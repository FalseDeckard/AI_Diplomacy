import logging
from typing import Dict
import asyncio

from .game_history import GameHistory
from .agent import DiplomacyAgent

logger = logging.getLogger(__name__)


async def planning_phase(
    game,
    agents: Dict[str, DiplomacyAgent],
    game_history: GameHistory,
    model_error_stats,
    log_file_path: str,
):
    """
    Lets each power generate a strategic plan using their DiplomacyAgent.
    """
    logger.info(f"Starting planning phase for {game.current_short_phase}...")
    active_powers = [p_name for p_name, p_obj in game.powers.items() if not p_obj.is_eliminated()]
    eliminated_powers = [p_name for p_name, p_obj in game.powers.items() if p_obj.is_eliminated()]

    logger.info(f"Active powers for planning: {active_powers}")
    if eliminated_powers:
        logger.info(f"Eliminated powers (skipped): {eliminated_powers}")
    else:
        logger.info("No eliminated powers yet.")

    board_state = game.get_state()

    planning_tasks = []
    power_order = []
    for power_name in active_powers:
        agent = agents.get(power_name)
        if agent is None:
            logger.warning(f"Agent for {power_name} not found in planning phase. Skipping.")
            continue

        planning_tasks.append(
            asyncio.create_task(
                agent.client.get_plan(
                    game,
                    board_state,
                    power_name,
                    game_history,
                    log_file_path,
                    agent_goals=agent.goals,
                    agent_relationships=agent.relationships,
                    agent_private_diary_str=agent.format_private_diary_for_prompt(),
                )
            )
        )
        power_order.append(power_name)
        logger.debug(f"Submitted async get_plan task for {power_name}.")

    logger.info(f"Waiting for {len(planning_tasks)} planning results...")
    results = await asyncio.gather(*planning_tasks, return_exceptions=True)

    for power_name, result in zip(power_order, results):
        if isinstance(result, Exception):
            logger.error(f"Exception during planning result processing for {power_name}: {result}")
            if power_name in model_error_stats:
                model_error_stats[power_name].setdefault("planning_execution_errors", 0)
                model_error_stats[power_name]["planning_execution_errors"] += 1
            else:
                model_error_stats.setdefault(f"{power_name}_planning_execution_errors", 0)
                model_error_stats[f"{power_name}_planning_execution_errors"] += 1
            continue

        plan_result = result or ""
        logger.info(f"Received planning result from {power_name}.")

        if isinstance(plan_result, str) and plan_result.startswith("Error:"):
            logger.warning(f"Agent {power_name} reported an error during planning: {plan_result}")
            if power_name in model_error_stats:
                model_error_stats[power_name].setdefault("planning_generation_errors", 0)
                model_error_stats[power_name]["planning_generation_errors"] += 1
            else:
                model_error_stats.setdefault(f"{power_name}_planning_generation_errors", 0)
                model_error_stats[f"{power_name}_planning_generation_errors"] += 1
        elif plan_result:
            agent = agents[power_name]
            agent.add_journal_entry(f"Generated plan for {game.current_short_phase}: {plan_result[:100]}...")
            game_history.add_plan(game.current_short_phase, power_name, plan_result)
            logger.debug(f"Added plan for {power_name} to history.")
        else:
            logger.warning(f"Agent {power_name} returned an empty plan.")

    logger.info("Planning phase processing complete.")
    return game_history
