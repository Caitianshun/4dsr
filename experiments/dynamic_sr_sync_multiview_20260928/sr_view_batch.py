"""Observation-only SR microbatch: no parameters, renderer, optimizer or RNG."""
MODES = {'single': ('J_joint', 1), 'async2': ('Async2', 2), 'sync2': ('Sync2', 2)}
ARM_TO_MODE = {arm: mode for mode, (arm, _) in MODES.items()}


class SRViewBatch:
    def __init__(self, mode='single', table=None, table_sha=None, cursor=6000):
        if mode not in MODES:
            raise ValueError(mode)
        if mode != 'single':
            assert table is not None and table_sha and len(table['rows'][mode]) == 6000
        self.mode, self.table, self.table_sha, self.cursor = mode, table, table_sha, cursor
        assert 6000 <= cursor <= 12000

    def observations(self, step, lr_index, sr_a_index):
        assert step == self.cursor + 1
        if self.mode == 'single':
            slots = ((sr_a_index, 0.1),)
        else:
            row = self.table['rows'][self.mode][step - 6001]
            assert (row['step'], row['lr_index'], row['sr_a_index']) == (step, lr_index, sr_a_index)
            slots = ((sr_a_index, 0.05), (row['sr_b_index'], 0.05))
        self.cursor = step
        return slots

    def state(self):
        slots = MODES[self.mode][1]
        return dict(module='SRViewBatch', mode=self.mode, slots=slots,
                    coefficients=[0.1 / slots] * slots, pair_sha256=self.table_sha,
                    cursor=self.cursor, adds_learnable_parameters=0, inference_operations=0)

    def validate_resume(self, state):
        for key, value in self.state().items():
            assert state[key] == value, ('SRViewBatch resume identity', key)

