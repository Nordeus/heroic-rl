import logging

TRAIN_CFG_FILENAME = "training_cfg.yml"
logger = logging.getLogger(__name__)


def run(cfg, is_resume=False):
    """Runs the experiment in an isolated Python runtime."""

    def thunk():
        """This method is going to be serialized and will run in an isolated runtime."""
        import os
        import os.path as osp

        import tensorflow as tf

        tf.compat.v1.disable_v2_behavior()
        # Turn off TF log spam.
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        import numpy as np

        # Register environment.
        from heroic_rl import gym_heroic  # noqa: F401

        from heroic_rl.agent import Agent, AgentSaver
        from heroic_rl.train import setup_logging

        from heroic_rl.utils.mpi_tools import mpi_fork, proc_id

        # Training with same seed should not be allowed at the same path - can
        # cause agent state to be restored at start of training.
        train_cfg_path = osp.join(cfg.output_dir, TRAIN_CFG_FILENAME)
        if proc_id() == 0:
            if not is_resume and osp.exists(train_cfg_path):
                raise ValueError("Training path already exists: %s" % train_cfg_path)

        # Forked code starts after fork call below.
        mpi_fork(cfg.cpus)

        setup_logging(cfg.output_dir)

        if proc_id() == 0:
            # Save training config for restoring.
            cfg.save(train_cfg_path)
            logger.info("Training config saved to %s", train_cfg_path)

        if len(cfg.gpus) > 0:
            # Manually setup CUDA_VISIBLE_DEVICES so that each forked child
            # gets a single GPU.
            gpu_id_idx = proc_id() % len(cfg.gpus)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus[gpu_id_idx])
            logger.info(
                "Process %d: CUDA_VISIBLE_DEVICES=%s, NUM_GPUs=%d",
                proc_id(),
                os.environ["CUDA_VISIBLE_DEVICES"],
                len(tf.config.experimental.list_physical_devices("GPU")),
            )

        saver = AgentSaver(cfg.output_dir, frequency=cfg.save_frequency_epochs)
        agents = [Agent(cfg, saver) for _ in range(cfg.num_agents)]

        cfg.seed += 10000 * proc_id()
        tf.compat.v1.set_random_seed(cfg.seed)
        np.random.seed(cfg.seed)
        logger.info("CPU%d: Seeding random with seed=%d", proc_id(), cfg.seed)

        next_agent_idx = 0
        # Keep track of maximum finished epochs for any agent, so others can
        # keep up, if they fall behind.
        agent_epochs_max = 0
        while len(agents) > 0:
            # __N.B.__ it is important for the agents to be ran in the same
            # order when restoring from cold-start. This is necessary because
            # TF Python callbacks are initialized in that same order, and they
            # are serialized in the graph. Initialization order affects their
            # naming.
            agent = agents[next_agent_idx]

            restore_path = agent.sync_restore_path()
            try:
                with agent:
                    agent.run(agent_epochs_max, restore_path=restore_path)
                    agent_epochs_max = max(agent_epochs_max, agent.epochs)
            except Exception:
                logger.error(
                    "Unhandled exception while running %s", agent, exc_info=True
                )
                raise

            if agent.is_done:
                logger.info("Removing %s from experiment", agent)
                del agents[next_agent_idx]
            else:
                next_agent_idx = (next_agent_idx + 1) % len(agents)

        if proc_id() == 0:
            logger.info("Experiment done!")

    import os
    import sys
    import subprocess
    import os.path as osp
    import cloudpickle
    import base64
    import zlib

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()],
        format="--- %(levelname)s %(name)s: %(message)s",
    )

    # Prepare to launch a script to run the experiment.
    pickled_thunk = cloudpickle.dumps(thunk)
    encoded_thunk = base64.b64encode(zlib.compress(pickled_thunk)).decode("utf-8")

    entrypoint = osp.join(osp.abspath(osp.dirname(__file__)), "entrypoint.py")
    cmd = [sys.executable if sys.executable else "python3", entrypoint, encoded_thunk]

    logger.info("Running experiment")
    try:
        subprocess.check_call(cmd, env=os.environ)
        logger.info("Experiment finished successfully")
    except subprocess.CalledProcessError as cpe:
        logger.error("Experiment failed with exit code %d" % cpe.returncode)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Ctrl+C detected, experiment interrupted")
        sys.exit(2)
