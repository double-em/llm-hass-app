"""LLM-powered Home Assistant App."""

import appdaemon.plugins.hass.hassapi as hass


class LLMApp(hass.Hass):
    """Main LLM App class."""

    def initialize(self):
        """Initialize the app."""
        self.log("LLM App initialized")

    def say_hello(self):
        """Example action."""
        self.log("Hello from LLM App!")
