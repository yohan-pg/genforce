# python3.7
"""Contains the running controller for evaluation."""

import os.path
import time

from .base_controller import BaseController
from ..misc import format_time
from ..lpips import calculate_lpips_given_images

__all__ = ['LPIPSEvaluator']


class LPIPSEvaluator(BaseController):
    """Defines the running controller for evaluation.

    This controller is used to evalute the GAN model using LPIPS metric.

    NOTE: The controller is set to `LAST` priority by default.
    """

    def __init__(self, config):
        assert isinstance(config, dict)
        config.setdefault('priority', 'LAST')
        super().__init__(config)

        self.num = config.get('num', 50000)
        self.ignore_cache = config.get('ignore_cache', False)
        self.align_tf = config.get('align_tf', True)
        self.file = None

    def setup(self, runner):
        assert hasattr(runner, 'lpips')
        file_path = os.path.join(runner.work_dir, f'metric_lpips{self.num}.txt')
        if runner.rank == 0:
            self.file = open(file_path, 'w')

    def close(self, runner):
        if runner.rank == 0:
            self.file.close()

    #! TODO here
    def execute_after_iteration(self, runner):
        mode = runner.mode  # save runner mode.
        start_time = time.time()
        lpips_value = runner.calculate_lpips_given_images(self.num,
                               ignore_cache=self.ignore_cache,
                               align_tf=self.align_tf)
        duration_str = format_time(time.time() - start_time)
        log_str = (f'LPIPS: {lpips_value:.5f} at iter {runner.iter:06d} '
                   f'({runner.seen_img / 1000:.1f} kimg). ({duration_str})')
        runner.logger.info(log_str)
        if runner.rank == 0:
            date = time.strftime("%Y-%m-%d %H:%M:%S")
            self.file.write(f'[{date}] {log_str}\n')
            self.file.flush()
        runner.set_mode(mode)  # restore runner mode.
