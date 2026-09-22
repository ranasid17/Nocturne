from qusa.services.research_service import run_model_workflow
from qusa.storage.runs import RunRepository


def test_research_workflow_records_missing_data_failure(tmp_path):
    repository = RunRepository(tmp_path / "history.sqlite3")
    config = {
        "data": {"paths": {"processed_data_dir": str(tmp_path / "processed"), "figures_dir": str(tmp_path / "figures")}},
        "model": {"output": {"model_output_path": str(tmp_path / "models")}, "parameters": {}},
        "backtest": {"volatility_filter": {"enabled": False}},
        "pipeline": {},
    }
    result = run_model_workflow("UPRO", config, repository=repository)
    assert result["success"] is False
    assert result["phases"]["training"]["status"] == "failed"
    assert repository.get_run(result["run_id"])["status"] == "failed"


def test_research_workflow_honors_phase_skips(tmp_path):
    repository = RunRepository(tmp_path / "history.sqlite3")
    config = {
        "data": {"paths": {"processed_data_dir": str(tmp_path / "processed"), "figures_dir": str(tmp_path / "figures")}},
        "model": {"output": {"model_output_path": str(tmp_path / "models")}, "parameters": {}},
        "backtest": {"volatility_filter": {"enabled": False}},
        "pipeline": {"skip_training": True, "skip_evaluation": True, "skip_backtest": True},
    }
    result = run_model_workflow("UPRO", config, repository=repository)
    assert result["success"] is True
    assert {phase["status"] for phase in result["phases"].values()} == {"skipped"}
