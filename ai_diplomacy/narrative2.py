"""Generate engaging narrative summaries and transparently patch the Diplomacy
Game engine to use them.

Usage: simply import `ai_diplomacy.narrative` *before* the game loop starts
(e.g. at the top of `lm_game.py`).  Import side-effects monkey-patch
`diplomacy.engine.game.Game._generate_phase_summary` so that:

1. The original (statistical) summary logic still runs.
2. The returned text is stored in `GamePhaseData.statistical_summary`.
3. A short narrative is produced via OpenAI `o3` and saved as the main
   `.summary`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from diplomacy.engine.game import Game

# Import to get model configuration and client loading
from .utils import get_special_models
from .clients import load_model_client
from config import config

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPECIAL_MODELS = get_special_models()
OPENAI_MODEL = SPECIAL_MODELS["phase_summary"]


# ---------------------------------------------------------------------------
# Helper to call the model synchronously
# ---------------------------------------------------------------------------


async def _call_model_async(statistical_summary: str, phase_key: str) -> str:
    """Return a 2–4 sentence spectator-friendly narrative using async client."""
    try:
        # Load the narrative client
        narrative_client = load_model_client(OPENAI_MODEL)

        system = (
            "You are an energetic e-sports commentator narrating a game of Diplomacy. "
            "Turn the provided phase recap into a concise, thrilling story (max 4 sentences). "
            "Highlight pivotal moves, supply-center swings, betrayals, and momentum shifts."
        )
        narrative_client.set_system_prompt(system)

        user = f"PHASE {phase_key}\n\nSTATISTICAL SUMMARY:\n{statistical_summary}\n\nNow narrate this phase for spectators."

        # Use the client's generate_response method
        response = await narrative_client.generate_response(
            prompt=user,
            temperature=0.7  # Some creativity for narrative
            #inject_random_seed=False,  # No need for random seed in narratives
        )

        return response.strip() if response else "(Narrative generation failed - empty response)"

    except Exception as e:  # Broad – we only log and degrade gracefully
        if config.ALLOW_NARATION_FAILURE:
            LOGGER.error(f"Narrative generation failed: {e}", exc_info=True)
            return "(Narrative generation failed)"
        else:
            raise e


async def _call_openai_async(statistical_summary: str, phase_key: str) -> str:
    """Return a 2–4 sentence spectator-friendly narrative."""
    return await _call_model_async(statistical_summary, phase_key)


def _call_openai(statistical_summary: str, phase_key: str) -> str:
    """Synchronous wrapper for backward compatibility."""
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        # Use asyncio.create_task to run in current loop
        import concurrent.futures
        import threading
        
        result = None
        exception = None
        
        def run_async():
            nonlocal result, exception
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result = new_loop.run_until_complete(_call_model_async(statistical_summary, phase_key))
                new_loop.close()
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=run_async)
        thread.start()
        thread.join()
        
        if exception:
            raise exception
        return result
        
    except RuntimeError:
        # No running loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_call_model_async(statistical_summary, phase_key))
        loop.close()
        return result


# ---------------------------------------------------------------------------
# Patch _generate_phase_summary
# ---------------------------------------------------------------------------

_original_gps: Callable = Game._generate_phase_summary  # type: ignore[attr-defined]


def _default_phase_summary_callback(self: Game, phase_key: str) -> Callable:
    """Generate a default statistical summary callback when none is provided."""

    def phase_summary_callback(system_prompt, user_prompt):
        # Get the current short phase for accessing game history
        current_short_phase = phase_key

        # 1) Gather the current board state, sorted by # of centers
        power_info = []
        for power_name, power in self.powers.items():
            units_list = list(power.units)
            centers_list = list(power.centers)
            power_info.append((power_name, len(centers_list), units_list, centers_list))
        # Sort by descending # of centers
        power_info.sort(key=lambda x: x[1], reverse=True)

        # 2) Build text lines for the top "Board State Overview"
        top_lines = ["Current Board State (Ordered by SC Count):"]
        for p_name, sc_count, units, centers in power_info:
            top_lines.append(f" • {p_name}: {sc_count} centers (needs 18 to win). Units={units} Centers={centers}")

        # 3) Map orders to "successful", "failed", or "other" outcomes
        success_dict = {}
        fail_dict = {}
        other_dict = {}

        orders_dict = self.order_history.get(current_short_phase, {})
        results_for_phase = self.result_history.get(current_short_phase, {})

        for pwr, pwr_orders in orders_dict.items():
            for order_str in pwr_orders:
                # Extract the unit from the string
                tokens = order_str.split()
                if len(tokens) < 3:
                    # Something malformed
                    other_dict.setdefault(pwr, []).append(order_str)
                    continue
                unit_name = " ".join(tokens[:2])
                # We retrieve the order results for that unit
                results_list = results_for_phase.get(unit_name, [])
                # Check if the results contain e.g. "dislodged", "bounce", "void"
                # We consider success if the result list is empty or has no negative results
                if not results_list or all(res not in ["bounce", "void", "no convoy", "cut", "dislodged", "disrupted"] for res in results_list):
                    success_dict.setdefault(pwr, []).append(order_str)
                elif any(res in ["bounce", "void", "no convoy", "cut", "dislodged", "disrupted"] for res in results_list):
                    fail_dict.setdefault(pwr, []).append(f"{order_str} - {', '.join(str(r) for r in results_list)}")
                else:
                    other_dict.setdefault(pwr, []).append(order_str)

        # 4) Build textual lists of successful, failed, and "other" moves
        def format_moves_dict(title, moves_dict):
            lines = [title]
            if not moves_dict:
                lines.append("  None.")
                return "\n".join(lines)
            for pwr in sorted(moves_dict.keys()):
                lines.append(f"  {pwr}:")
                for mv in moves_dict[pwr]:
                    lines.append(f"    {mv}")
            return "\n".join(lines)

        success_section = format_moves_dict("Successful Moves:", success_dict)
        fail_section = format_moves_dict("Unsuccessful Moves:", fail_dict)
        other_section = format_moves_dict("Other / Unclassified Moves:", other_dict)

        # 5) Combine everything into the final summary text
        summary_parts = []
        summary_parts.append("\n".join(top_lines))
        summary_parts.append("\n" + success_section)
        summary_parts.append("\n" + fail_section)

        # Only include "Other" section if it has content
        if other_dict:
            summary_parts.append("\n" + other_section)

        return f"Phase {current_short_phase} Summary:\n\n" + "\n".join(summary_parts)

    return phase_summary_callback


def _patched_generate_phase_summary(self: Game, phase_key, summary_callback=None):  # type: ignore[override]
    # If no callback provided, use our default one
    if summary_callback is None:
        summary_callback = _default_phase_summary_callback(self, phase_key)

    # 1) Call original implementation → statistical summary
    statistical = _original_gps(self, phase_key, summary_callback)
    LOGGER.debug(f"[{phase_key}] Original summary returned: {statistical!r}")

    # 2) Persist statistical summary separately
    phase_data = None
    try:
        phase_data = self.get_phase_from_history(str(phase_key))
        if hasattr(phase_data, "statistical_summary"):
            LOGGER.debug(f"[{phase_key}] Assigning to phase_data.statistical_summary: {statistical!r}")
            phase_data.statistical_summary = statistical  # type: ignore[attr-defined]
        else:
            LOGGER.warning(f"[{phase_key}] phase_data object does not have attribute 'statistical_summary'. Type: {type(phase_data)}")
    except Exception as exc:
        LOGGER.warning("Could not retrieve phase_data or store statistical_summary for %s: %s", phase_key, exc)

    # 3) Generate narrative summary
    narrative = _call_openai(statistical, phase_key)

    # 4) Save narrative as the canonical summary
    try:
        if phase_data:
            phase_data.summary = narrative  # type: ignore[attr-defined]
            self.phase_summaries[str(phase_key)] = narrative  # type: ignore[attr-defined]
            LOGGER.debug(f"[{phase_key}] Narrative summary stored successfully.")
        else:
            LOGGER.warning(f"[{phase_key}] Cannot store narrative summary because phase_data is None.")
    except Exception as exc:
        LOGGER.warning("Could not store narrative summary for %s: %s", phase_key, exc)

    return narrative


# Monkey-patch
Game._generate_phase_summary = _patched_generate_phase_summary  # type: ignore[assignment]

LOGGER.info("Game._generate_phase_summary patched with narrative generation.")
