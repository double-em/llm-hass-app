"""Home Assistant config flow for LLM AI Dashboard integration."""

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback

from .const import (
    DOMAIN,
    CONF_DEFAULT_PROVIDER,
    CONF_DEFAULT_VOICE_SPEED,
    CONF_DEFAULT_DIFFUSION_STEPS,
    DEFAULT_PROVIDER,
    DEFAULT_VOICE_SPEED,
    DEFAULT_DIFFUSION_STEPS,
)


@config_entries.HANDLERS.register(DOMAIN)
class LLMAIConfigFlow(config_entries.ConfigFlowHandler):
    """Config flow for LLM AI Dashboard."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle a flow initialized by the user."""
        errors = {}

        if user_input is not None:
            return self.async_create_entry(title="LLM AI Dashboard", data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Optional(CONF_DEFAULT_PROVIDER, default=DEFAULT_PROVIDER): str,
                vol.Optional(CONF_DEFAULT_VOICE_SPEED, default=DEFAULT_VOICE_SPEED): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=2.0)),
                vol.Optional(CONF_DEFAULT_DIFFUSION_STEPS, default=DEFAULT_DIFFUSION_STEPS): vol.All(vol.Coerce(int), vol.Range(min=1, max=100)),
            }),
            errors=errors,
            description_placeholders={
                "provider_desc": "Default AI provider to use (e.g., minimax)",
                "speed_desc": "Voice playback speed (0.5x to 2.0x)",
                "steps_desc": "Diffusion steps for voice generation (1-100)",
            },
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get options flow handler."""
        return LLMAIOptionsFlow(config_entry)


class LLMAIOptionsFlow(config_entries.OptionsFlow):
    """Options flow for LLM AI Dashboard."""

    async def async_step_init(self, user_input=None):
        """Manage options."""
        if user_input is not None:
            return self.async_create_entry(title="LLM AI Dashboard", data=user_input)

        options = self.config_entry.options or {}
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Optional(CONF_DEFAULT_PROVIDER, default=options.get(CONF_DEFAULT_PROVIDER, DEFAULT_PROVIDER)): str,
                vol.Optional(CONF_DEFAULT_VOICE_SPEED, default=options.get(CONF_DEFAULT_VOICE_SPEED, DEFAULT_VOICE_SPEED)): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=2.0)),
                vol.Optional(CONF_DEFAULT_DIFFUSION_STEPS, default=options.get(CONF_DEFAULT_DIFFUSION_STEPS, DEFAULT_DIFFUSION_STEPS)): vol.All(vol.Coerce(int), vol.Range(min=1, max=100)),
            }),
        )
