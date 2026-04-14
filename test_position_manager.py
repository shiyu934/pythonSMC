import sys
from unittest.mock import MagicMock, patch

# We MUST mock before importing position_manager because it imports agent, which imports torch
mock_torch = MagicMock()
mock_numpy = MagicMock()
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = MagicMock()
sys.modules["numpy"] = mock_numpy

import pytest
from datetime import datetime, timedelta
import os

# Now import the class to test
from position_manager import PositionManager

class TestPositionManager:
    @pytest.fixture
    def pm(self):
        return PositionManager(num_agents=2, mdd_limit=-5.0, dead_threshold=-10.0, cooldown=300)

    @pytest.fixture
    def mock_arena(self):
        arena = MagicMock()
        # Default behavior: return the difference as PnL for simplicity in some tests
        arena.calculate_reward.side_effect = lambda entry, now, action: (now - entry) if action == 1 else (entry - now)
        return arena

    def test_init(self, pm):
        assert pm.mdd_limit == -5.0
        assert pm.dead_threshold == -10.0
        assert pm.cooldown == 300
        assert len(pm.stats) == 2
        assert pm.stats["0"] == {"pnl": 0.0, "trades": 0}
        assert pm.locks["0"]["current_action"] == 0
        assert pm.entry_prices["0"] == 0.0

    def test_get_safe_pnl_no_action(self, pm, mock_arena):
        assert pm._get_safe_pnl(100.0, 110.0, 0, mock_arena) == 0.0

    def test_get_safe_pnl_no_entry(self, pm, mock_arena):
        assert pm._get_safe_pnl(0.0, 110.0, 1, mock_arena) == 0.0

    def test_get_safe_pnl_normal(self, pm, mock_arena):
        mock_arena.calculate_reward.return_value = 10.0
        assert pm._get_safe_pnl(100.0, 110.0, 1, mock_arena) == 10.0

    def test_get_safe_pnl_capped(self, pm, mock_arena):
        mock_arena.calculate_reward.side_effect = None
        mock_arena.calculate_reward.return_value = -150.0
        assert pm._get_safe_pnl(100.0, 50.0, 1, mock_arena) == -98.0

    @patch("os.path.exists")
    @patch("os.remove")
    def test_rebirth(self, mock_remove, mock_exists, pm):
        mock_exists.return_value = True
        a_id = "0"
        pm.stats[a_id] = {"pnl": -15.0, "trades": 5}
        pm.entry_prices[a_id] = 100.0
        pm.locks[a_id]["current_action"] = 1

        pm._rebirth(a_id)

        mock_remove.assert_called_once_with(f"./models/agent_{a_id}.pth")
        assert pm.stats[a_id] == {"pnl": 0.0, "trades": 0}
        assert pm.entry_prices[a_id] == 0.0
        assert pm.locks[a_id]["current_action"] == 0

    @patch("position_manager.TradingAgent")
    def test_check_and_execute_priority1_death(self, mock_agent_class, pm, mock_arena):
        a_id = 0
        pm.stats[str(a_id)]["pnl"] = -9.0
        mock_arena.calculate_reward.side_effect = None
        mock_arena.calculate_reward.return_value = -2.0 # Total -11.0 < -10.0
        pm.entry_prices[str(a_id)] = 100.0
        pm.locks[str(a_id)]["current_action"] = 1

        with patch.object(pm, "_rebirth") as mock_rebirth:
            action, status, new_agent = pm.check_and_execute(a_id, None, 98.0, 1, mock_arena, 64)

            mock_rebirth.assert_called_once_with(str(a_id))
            assert action == 0
            assert status == "🆕NEW"
            assert new_agent == mock_agent_class.return_value
            mock_agent_class.assert_called_once_with(agent_id=str(a_id), state_dim=64)

    def test_check_and_execute_priority2_stop_loss(self, pm, mock_arena):
        a_id = 0
        pm.entry_prices[str(a_id)] = 100.0
        pm.locks[str(a_id)]["current_action"] = 1
        mock_arena.calculate_reward.side_effect = None
        mock_arena.calculate_reward.return_value = -6.0 # < -5.0 (mdd_limit)

        action, status, new_agent = pm.check_and_execute(a_id, None, 94.0, 1, mock_arena, 64)

        assert action == 0
        assert status == "🛑STOP"
        assert pm.stats[str(a_id)]["pnl"] == -6.0
        assert pm.stats[str(a_id)]["trades"] == 1
        assert pm.entry_prices[str(a_id)] == 0.0
        assert pm.locks[str(a_id)]["current_action"] == 0
        assert new_agent is None

    def test_check_and_execute_priority3_lock(self, pm, mock_arena):
        a_id = 0
        pm.locks[str(a_id)]["last_change_time"] = datetime.now()
        pm.locks[str(a_id)]["current_action"] = 1

        # Try to change action to 2
        action, status, new_agent = pm.check_and_execute(a_id, None, 100.0, 2, mock_arena, 64)

        assert action == 1
        assert status == "⏳LOCK"
        assert new_agent is None

    def test_check_and_execute_priority4_open_position(self, pm, mock_arena):
        a_id = 0
        pm.locks[str(a_id)]["current_action"] = 0
        pm.locks[str(a_id)]["last_change_time"] = datetime.now() - timedelta(seconds=400)

        action, status, new_agent = pm.check_and_execute(a_id, None, 100.0, 1, mock_arena, 64)

        assert action == 1
        assert status == "⚡ACT-1"
        assert pm.entry_prices[str(a_id)] == 100.0
        assert pm.locks[str(a_id)]["current_action"] == 1
        assert new_agent is None

    def test_check_and_execute_priority4_close_position(self, pm, mock_arena):
        a_id = 0
        pm.entry_prices[str(a_id)] = 100.0
        pm.locks[str(a_id)]["current_action"] = 1
        pm.locks[str(a_id)]["last_change_time"] = datetime.now() - timedelta(seconds=400)
        mock_arena.calculate_reward.side_effect = None
        mock_arena.calculate_reward.return_value = 2.0

        action, status, new_agent = pm.check_and_execute(a_id, None, 102.0, 0, mock_arena, 64)

        assert action == 0
        assert status == "⚡ACT-0"
        assert pm.stats[str(a_id)]["pnl"] == 2.0
        assert pm.stats[str(a_id)]["trades"] == 1
        assert pm.entry_prices[str(a_id)] == 0.0
        assert pm.locks[str(a_id)]["current_action"] == 0
        assert new_agent is None

    def test_check_and_execute_priority4_switch_position(self, pm, mock_arena):
        a_id = 0
        pm.entry_prices[str(a_id)] = 100.0
        pm.locks[str(a_id)]["current_action"] = 1
        pm.locks[str(a_id)]["last_change_time"] = datetime.now() - timedelta(seconds=400)
        mock_arena.calculate_reward.side_effect = None
        mock_arena.calculate_reward.return_value = 2.0

        # Switch from 1 to 2
        action, status, new_agent = pm.check_and_execute(a_id, None, 102.0, 2, mock_arena, 64)

        assert action == 2
        assert status == "⚡ACT-2"
        assert pm.stats[str(a_id)]["pnl"] == 2.0
        assert pm.stats[str(a_id)]["trades"] == 1
        assert pm.entry_prices[str(a_id)] == 102.0
        assert pm.locks[str(a_id)]["current_action"] == 2
        assert new_agent is None

    def test_check_and_execute_hold(self, pm, mock_arena):
        a_id = 0
        pm.locks[str(a_id)]["current_action"] = 1
        pm.entry_prices[str(a_id)] = 100.0
        mock_arena.calculate_reward.side_effect = None
        mock_arena.calculate_reward.return_value = 1.0

        action, status, new_agent = pm.check_and_execute(a_id, None, 101.0, 1, mock_arena, 64)

        assert action == 1
        assert status == "HOLD"
        assert new_agent is None

    def test_check_and_execute_idle(self, pm, mock_arena):
        a_id = 0
        pm.locks[str(a_id)]["current_action"] = 0

        action, status, new_agent = pm.check_and_execute(a_id, None, 100.0, 0, mock_arena, 64)

        assert action == 0
        assert status == "IDLE"
        assert new_agent is None
