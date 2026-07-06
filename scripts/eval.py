from __future__ import annotations

import hydra
from omegaconf import DictConfig

from docr.evaluation.metrics import ocr_metrics
from docr.evaluation.report import write_json_report, write_markdown_report


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    prediction = cfg.get("prediction", "")
    target = cfg.get("target", "")
    metrics = ocr_metrics(str(prediction), str(target))
    write_json_report(cfg.eval.json_path, metrics)
    write_markdown_report(cfg.eval.report_path, metrics)
    print(metrics)


if __name__ == "__main__":
    main()

