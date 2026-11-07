from pathlib import Path

from airflow.dag_processing.dagbag import DagBag


project_root = Path(__file__).resolve().parents[1]
dag_folder = project_root / "airflow" / "dags"

dag_bag = DagBag(dag_folder=str(dag_folder))

assert not dag_bag.import_errors, dag_bag.import_errors
assert "cmapss_data_prediction" in dag_bag.dags

print("DAG import check passed.")