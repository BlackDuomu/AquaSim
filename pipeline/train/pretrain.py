#!/usr/bin/env python
from __future__ import annotations

from _common import add_common_args, build_pretrain_argv, load_yaml, print_dry_run, project_path, run_trainer, validate_paths


def main() -> None:
    args = add_common_args("AquaSim global pretrain entrypoint.").parse_args()
    cfg = load_yaml(project_path(args.config))
    argv = build_pretrain_argv(cfg)
    missing = validate_paths(cfg["paths"], ["unified_dir", "global_pretrain_patch_dir"])
    if args.dry_run:
        print_dry_run("pretrain", args.config, argv, missing)
        return
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    run_trainer(argv)


if __name__ == "__main__":
    main()
