"""Load tests for concurrent TTS and vector memory operations.

This module tests the system's ability to handle concurrent requests
under load, including:
- Concurrent TTS generation
- Concurrent vector embedding operations
- Voice cache hit/miss scenarios under load

Run with: pytest llm_app/tests/test_load_tts.py -v
Or for a quick smoke test: pytest llm_app/tests/test_load_tts.py -v -k "smoke"
"""

import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pytest

# Set testing environment before imports
os.environ["TESTING"] = "true"

from memory import VectorStore, EmbeddingEngine
from memory.ha_assists import HAAssistsClient, HAAssistsConfig
from llm_app.voice_cache import VoiceLineCache
from llm_app.voice_cache_store import VoiceCacheStore


# Test configuration
MAX_WORKERS = 10  # Number of concurrent threads
TTS_REQUEST_COUNT = 20  # Number of TTS requests to simulate
EMBEDDING_BATCH_SIZE = 10  # Batch size for embedding operations


@dataclass
class LoadTestResult:
    """Results from a load test run."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_ms: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    errors: List[str]


class LoadTestRunner:
    """Helper class to run load tests and collect metrics."""

    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.results: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def run_concurrent_tasks(self, task_fn, task_args: List[tuple], description: str) -> LoadTestResult:
        """Run tasks concurrently and collect metrics.

        Args:
            task_fn: Function to execute
            task_args: List of argument tuples for each task
            description: Description of the test

        Returns:
            LoadTestResult with metrics
        """
        self.results = []
        self.errors = []

        start_time = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(task_fn, *args) for args in task_args]

            for future in concurrent.futures.as_completed(futures):
                task_start = time.perf_counter()
                try:
                    result = future.result()
                    task_duration = (time.perf_counter() - task_start) * 1000  # ms
                    self.results.append({
                        "success": True,
                        "duration_ms": task_duration,
                        "result": result
                    })
                except Exception as e:
                    task_duration = (time.perf_counter() - task_start) * 1000
                    self.errors.append(str(e))
                    self.results.append({
                        "success": False,
                        "duration_ms": task_duration,
                        "error": str(e)
                    })

        end_time = time.perf_counter()
        total_duration_ms = (end_time - start_time) * 1000

        successful = sum(1 for r in self.results if r["success"])
        failed = sum(1 for r in self.results if not r["success"])
        latencies = sorted([r["duration_ms"] for r in self.results])

        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)

        return LoadTestResult(
            total_requests=len(task_args),
            successful_requests=successful,
            failed_requests=failed,
            total_duration_ms=total_duration_ms,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            p95_latency_ms=latencies[p95_idx] if latencies else 0,
            p99_latency_ms=latencies[p99_idx] if latencies else 0,
            throughput_rps=(len(task_args) / (total_duration_ms / 1000)) if total_duration_ms > 0 else 0,
            errors=self.errors
        )


class TestConcurrentVoiceCache:
    """Test concurrent voice cache operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_store = VoiceCacheStore(data_dir=self.temp_dir)

    def _generate_mock_audio(self, text: str) -> bytes:
        """Generate mock audio bytes for testing."""
        return f"mock_audio_for_{text}".encode()[:1024]

    def _simulate_tts_request(self, text: str, voice_name: str = "test-voice") -> Dict[str, Any]:
        """Simulate a TTS request with caching logic.

        This simulates the TTS flow:
        1. Check cache
        2. If miss, generate audio (mocked)
        3. Store in cache
        """
        # Build cache key
        key_parts = [text, "1.0", "32", "auto"]
        if voice_name:
            key_parts.append(f"voice:{voice_name}")
        cache_key = hashlib.sha256("|".join(key_parts).encode()).hexdigest()

        # Check cache
        cached = self.cache_store.get(cache_key, include_audio=False)
        if cached:
            return {
                "cached": True,
                "text": text,
                "text_hash": cache_key
            }

        # Simulate TTS generation
        audio = self._generate_mock_audio(text)

        # Save to cache
        self.cache_store.save(
            text_hash=cache_key,
            text=text,
            audio_blob=audio,
            provider="omnivoice",
            voice_name=voice_name,
            speed=1.0,
            num_steps=32,
            audio_format="wav",
            sample_rate=16000,
            duration_ms=1500,
        )

        return {
            "cached": False,
            "text": text,
            "text_hash": cache_key,
            "generated": True
        }

    def test_concurrent_cache_writes(self):
        """Test concurrent cache write operations."""
        runner = LoadTestRunner(max_workers=MAX_WORKERS)

        # Generate unique texts for each request
        texts = [f"Test text number {i} for concurrent cache write" for i in range(TTS_REQUEST_COUNT)]
        task_args = [(text,) for text in texts]

        result = runner.run_concurrent_tasks(self._simulate_tts_request, task_args, "cache_write")

        # All requests should succeed
        assert result.successful_requests == TTS_REQUEST_COUNT, f"Expected all {TTS_REQUEST_COUNT} to succeed, got {result.successful_requests}"
        assert result.failed_requests == 0, f"Expected 0 failures, got {result.failed_requests}"

        # Verify cache has entries
        stats = self.cache_store.get_stats()
        assert stats["total_entries"] == TTS_REQUEST_COUNT

        print(f"\n[Concurrent Cache Writes] Results:")
        print(f"  Total requests: {result.total_requests}")
        print(f"  Successful: {result.successful_requests}")
        print(f"  Avg latency: {result.avg_latency_ms:.2f}ms")
        print(f"  P95 latency: {result.p95_latency_ms:.2f}ms")
        print(f"  P99 latency: {result.p99_latency_ms:.2f}ms")
        print(f"  Throughput: {result.throughput_rps:.2f} req/s")

    def test_concurrent_cache_hits(self):
        """Test concurrent cache hit scenarios (reads)."""
        # Pre-populate cache
        texts = [f"Pre-cached text {i}" for i in range(10)]
        for text in texts:
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            self.cache_store.save(
                text_hash=cache_key,
                text=text,
                audio_blob=b"mock_audio",
                provider="omnivoice",
                speed=1.0,
            )

        runner = LoadTestRunner(max_workers=MAX_WORKERS)

        # All requests should hit cache
        task_args = [(text,) for text in texts * 2]  # Hit each twice
        result = runner.run_concurrent_tasks(self._simulate_tts_request, task_args, "cache_hit")

        assert result.successful_requests == len(task_args)
        # All should be cache hits
        cache_hits = sum(1 for r in runner.results if r["success"] and r["result"].get("cached"))
        assert cache_hits == len(task_args), f"Expected all {len(task_args)} to be cache hits"

        print(f"\n[Concurrent Cache Hits] Results:")
        print(f"  Total requests: {result.total_requests}")
        print(f"  Cache hits: {cache_hits}")
        print(f"  Avg latency: {result.avg_latency_ms:.2f}ms")

    def test_concurrent_mixed_cache_operations(self):
        """Test mixed cache hit/miss scenarios under concurrent load."""
        # Pre-cache half the texts
        pre_cached = [f"Pre-cached {i}" for i in range(5)]
        uncached = [f"Uncached {i}" for i in range(5)]
        all_texts = pre_cached + uncached

        for text in pre_cached:
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            self.cache_store.save(
                text_hash=cache_key,
                text=text,
                audio_blob=b"mock_audio",
                provider="omnivoice",
            )

        runner = LoadTestRunner(max_workers=MAX_WORKERS)
        task_args = [(text,) for text in all_texts]
        result = runner.run_concurrent_tasks(self._simulate_tts_request, task_args, "mixed_ops")

        assert result.successful_requests == len(task_args)

        cache_hits = sum(1 for r in runner.results if r["success"] and r["result"].get("cached"))
        cache_misses = sum(1 for r in runner.results if r["success"] and not r["result"].get("cached"))

        assert cache_hits == len(pre_cached), f"Expected {len(pre_cached)} cache hits, got {cache_hits}"
        assert cache_misses == len(uncached), f"Expected {len(uncached)} cache misses, got {cache_misses}"

        print(f"\n[Mixed Cache Operations] Results:")
        print(f"  Cache hits: {cache_hits}")
        print(f"  Cache misses: {cache_misses}")


class TestConcurrentVectorOperations:
    """Test concurrent vector store operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(data_dir=self.temp_dir)
        self.embedding_engine = EmbeddingEngine()

    @patch.object(VectorStore, '_get_chroma_client')
    def test_concurrent_vector_adds(self, mock_chroma):
        """Test concurrent vector memory additions."""
        mock_collection = MagicMock()
        mock_chroma.return_value = MagicMock(get_or_create_collection=MagicMock(return_value=mock_collection))

        runner = LoadTestRunner(max_workers=MAX_WORKERS)

        def add_memory(index: int) -> Dict[str, Any]:
            content = f"Test memory content number {index}"
            embedding = self.embedding_engine.encode(content)
            entry = self.vector_store.add_entry(
                content=content,
                embedding=embedding,
                tags=["load-test", f"index-{index}"],
                source="load-test"
            )
            return entry

        task_args = [(i,) for i in range(TTS_REQUEST_COUNT)]
        result = runner.run_concurrent_tasks(add_memory, task_args, "vector_add")

        assert result.successful_requests == TTS_REQUEST_COUNT

        print(f"\n[Concurrent Vector Adds] Results:")
        print(f"  Total requests: {result.total_requests}")
        print(f"  Avg latency: {result.avg_latency_ms:.2f}ms")
        print(f"  P95 latency: {result.p95_latency_ms:.2f}ms")
        print(f"  Throughput: {result.throughput_rps:.2f} req/s")

    @patch.object(VectorStore, '_get_chroma_client')
    def test_concurrent_vector_searches(self, mock_chroma):
        """Test concurrent vector searches."""
        mock_collection = MagicMock()
        mock_chroma.return_value = MagicMock(get_or_create_collection=MagicMock(return_value=mock_collection))

        # Pre-add some entries
        for i in range(EMBEDDING_BATCH_SIZE):
            content = f"Memory about topic {i} with some content"
            embedding = self.embedding_engine.encode(content)
            self.vector_store.add_entry(
                content=content,
                embedding=embedding,
                tags=["test"],
                source="load-test"
            )

        runner = LoadTestRunner(max_workers=MAX_WORKERS)

        queries = [
            "topic 1 content",
            "topic 2 information",
            "memory test",
            "topic 0 details",
            "some content here",
        ]

        def search_vector(query: str) -> Dict[str, Any]:
            embedding = self.embedding_engine.encode(query)
            results = self.vector_store.search(
                query_embedding=embedding,
                limit=5,
                threshold=0.5
            )
            return {"query": query, "results": len(results)}

        task_args = [(q,) for q in queries * 4]  # Run each query 4 times concurrently
        result = runner.run_concurrent_tasks(search_vector, task_args, "vector_search")

        assert result.successful_requests == len(task_args)

        print(f"\n[Concurrent Vector Searches] Results:")
        print(f"  Total requests: {result.total_requests}")
        print(f"  Avg latency: {result.avg_latency_ms:.2f}ms")
        print(f"  P95 latency: {result.p95_latency_ms:.2f}ms")


class TestConcurrentEmbeddingGeneration:
    """Test concurrent embedding generation operations."""

    def setup_method(self):
        """Reset embedding engine singleton."""
        EmbeddingEngine._instance = None
        EmbeddingEngine._model = None
        self.embedding_engine = EmbeddingEngine()

    def test_concurrent_embedding_encoding(self):
        """Test concurrent embedding generation."""
        runner = LoadTestRunner(max_workers=MAX_WORKERS)

        texts = [f"Text to embed number {i}" for i in range(TTS_REQUEST_COUNT)]

        def encode_text(text: str) -> Dict[str, Any]:
            embedding = self.embedding_engine.encode(text)
            return {"text": text, "dimensions": len(embedding)}

        task_args = [(t,) for t in texts]
        result = runner.run_concurrent_tasks(encode_text, task_args, "embedding_encode")

        assert result.successful_requests == TTS_REQUEST_COUNT

        # All embeddings should have same dimension
        for r in runner.results:
            if r["success"]:
                assert r["result"]["dimensions"] == 384, f"Expected 384 dimensions, got {r['result']['dimensions']}"

        print(f"\n[Concurrent Embedding Encoding] Results:")
        print(f"  Total requests: {result.total_requests}")
        print(f"  Avg latency: {result.avg_latency_ms:.2f}ms")
        print(f"  P95 latency: {result.p95_latency_ms:.2f}ms")
        print(f"  Throughput: {result.throughput_rps:.2f} req/s")

    def test_batch_embedding_encoding(self):
        """Test batch embedding generation (simulates batch TTS optimization)."""
        texts = [f"Batch text {i}" for i in range(EMBEDDING_BATCH_SIZE)]

        start = time.perf_counter()
        embeddings = self.embedding_engine.encode_batch(texts)
        duration_ms = (time.perf_counter() - start) * 1000

        assert len(embeddings) == EMBEDDING_BATCH_SIZE
        for emb in embeddings:
            assert len(emb) == 384

        print(f"\n[Batch Embedding Encoding] Results:")
        print(f"  Batch size: {EMBEDDING_BATCH_SIZE}")
        print(f"  Total duration: {duration_ms:.2f}ms")
        print(f"  Per-item avg: {duration_ms/EMBEDDING_BATCH_SIZE:.2f}ms")


class TestHAAssistsConcurrent:
    """Test concurrent HA Assists operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ha_client = HAAssistsClient(data_dir=self.temp_dir)

    @patch('requests.post')
    def test_concurrent_assist_requests(self, mock_post):
        """Test concurrent HA assist pipeline requests."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Response from HA"}
        mock_post.return_value = mock_response

        self.ha_client.update_config(
            enabled=True,
            ha_url="http://localhost:8123",
            ha_token="test_token"
        )

        runner = LoadTestRunner(max_workers=MAX_WORKERS)

        intents = [
            "Turn on the lights",
            "What's the weather?",
            "Set temperature to 72",
            "Lock the door",
            "Play some music",
        ]

        def send_assist_request(intent: str) -> Dict[str, Any]:
            result = self.ha_client.process_assist_pipeline(
                text=intent,
                conversation_id="test_conv",
                language="en"
            )
            return result

        task_args = [(intent,) for intent in intents * 4]  # 20 total requests
        result = runner.run_concurrent_tasks(send_assist_request, task_args, "ha_assist")

        assert result.successful_requests == len(task_args)
        assert mock_post.call_count == len(task_args)

        print(f"\n[Concurrent HA Assist] Results:")
        print(f"  Total requests: {result.total_requests}")
        print(f"  Avg latency: {result.avg_latency_ms:.2f}ms")
        print(f"  P95 latency: {result.p95_latency_ms:.2f}ms")
        print(f"  Throughput: {result.throughput_rps:.2f} req/s")


class TestCacheInvalidation:
    """Test cache invalidation under concurrent load."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_store = VoiceCacheStore(data_dir=self.temp_dir)

    def test_concurrent_invalidation(self):
        """Test concurrent cache invalidation."""
        # Pre-populate cache
        text_hashes = []
        for i in range(20):
            cache_key = hashlib.sha256(f"text_{i}".encode()).hexdigest()
            text_hashes.append(cache_key)
            self.cache_store.save(
                text_hash=cache_key,
                text=f"text_{i}",
                audio_blob=b"mock_audio",
                provider="omnivoice",
            )

        runner = LoadTestRunner(max_workers=MAX_WORKERS)

        def invalidate_cache(text_hash: str) -> bool:
            return self.cache_store.invalidate(text_hash)

        task_args = [(th,) for th in text_hashes]
        result = runner.run_concurrent_tasks(invalidate_cache, task_args, "cache_invalidate")

        assert result.successful_requests == len(text_hashes)

        # All should be invalid now
        stats = self.cache_store.get_stats()
        assert stats["total_entries"] == 0

        print(f"\n[Concurrent Cache Invalidation] Results:")
        print(f"  Total invalidations: {result.total_requests}")
        print(f"  Avg latency: {result.avg_latency_ms:.2f}ms")


class TestEndToEndConcurrent:
    """End-to-end concurrent load tests simulating real usage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(data_dir=self.temp_dir)
        self.cache_store = VoiceCacheStore(data_dir=self.temp_dir)
        self.ha_client = HAAssistsClient(data_dir=self.temp_dir)
        self.embedding_engine = EmbeddingEngine()

    @patch.object(VectorStore, '_get_chroma_client')
    @patch('requests.post')
    def test_full_concurrent_workflow(self, mock_post, mock_chroma):
        """Test full concurrent workflow: TTS + Vector + HA Assist."""
        mock_collection = MagicMock()
        mock_chroma.return_value = MagicMock(get_or_create_collection=MagicMock(return_value=mock_collection))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Processed successfully"}
        mock_post.return_value = mock_response

        self.ha_client.update_config(
            enabled=True,
            ha_url="http://localhost:8123",
            ha_token="test_token"
        )

        runner = LoadTestRunner(max_workers=5)

        def full_workflow(index: int) -> Dict[str, Any]:
            text = f"User query number {index}"

            # 1. Check cache for TTS
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            cached = self.cache_store.get(cache_key)
            if not cached:
                # Generate and cache
                self.cache_store.save(
                    text_hash=cache_key,
                    text=text,
                    audio_blob=b"mock_audio",
                    provider="omnivoice",
                )

            # 2. Store in vector memory
            embedding = self.embedding_engine.encode(text)
            self.vector_store.add_entry(
                content=text,
                embedding=embedding,
                tags=["query"],
                source="user-input"
            )

            # 3. Send to HA Assist
            assist_result = self.ha_client.process_assist_pipeline(text=text)

            return {
                "text": text,
                "cache_hit": cached is not None,
                "vector_stored": True,
                "assist_response": assist_result.get("text") if "text" in assist_result else None
            }

        task_args = [(i,) for i in range(10)]
        result = runner.run_concurrent_tasks(full_workflow, task_args, "full_workflow")

        assert result.successful_requests == 10

        print(f"\n[Full Concurrent Workflow] Results:")
        print(f"  Total workflows: {result.total_requests}")
        print(f"  Avg latency: {result.avg_latency_ms:.2f}ms")
        print(f"  P95 latency: {result.p95_latency_ms:.2f}ms")
        print(f"  Throughput: {result.throughput_rps:.2f} req/s")


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests for establishing baselines."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_store = VoiceCacheStore(data_dir=self.temp_dir)

    def test_cache_throughput_benchmark(self):
        """Benchmark cache write throughput."""
        texts = [f"Benchmark text {i}" for i in range(100)]

        start = time.perf_counter()
        for text in texts:
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            self.cache_store.save(
                text_hash=cache_key,
                text=text,
                audio_blob=b"x" * 1024,  # 1KB of audio
                provider="omnivoice",
            )
        duration = time.perf_counter() - start

        throughput = len(texts) / duration

        print(f"\n[Cache Write Benchmark]")
        print(f"  Operations: {len(texts)}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Throughput: {throughput:.2f} ops/s")

        assert throughput > 0, "Should achieve positive throughput"

    @pytest.mark.smoke
    def test_smoke_concurrent_tts(self):
        """Quick smoke test for concurrent TTS."""
        temp_dir = tempfile.mkdtemp()
        cache_store = VoiceCacheStore(data_dir=temp_dir)

        def mock_tts(i):
            cache_key = hashlib.sha256(f"smoke_test_{i}".encode()).hexdigest()
            cache_store.save(
                text_hash=cache_key,
                text=f"smoke_test_{i}",
                audio_blob=b"audio",
                provider="omnivoice",
            )
            return cache_key

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(mock_tts, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
