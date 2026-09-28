"""Protocol-bound four-task decisions; active and valid endpoints are never relaunched."""
def next_action(readiness, service_active, endpoint_valid, checkpoint9000_valid):
    if not readiness: return 'blocked_identity'
    if service_active: return 'wait_existing'
    if endpoint_valid: return 'evaluate_endpoint'
    if checkpoint9000_valid: return 'resume_9000'
    return 'start_parent'


def fixed_tasks(protocol):
    tasks = protocol['task_plan']
    assert [(t['repeat'], t['arm']) for t in tasks] == [
        ('1', 'Async2'), ('1', 'Sync2'), ('2', 'Sync2'), ('2', 'Async2')]
    assert all(t['start'] == 6000 and t['stop'] == 12000 and
               t['sr_slots'] == 2 and t['rgb_per_update'] == 3 for t in tasks)
    assert sum(t['effective_updates'] for t in tasks) == protocol['budget']['effective_formal_max']
    return tasks
