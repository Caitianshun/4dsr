"""Pure scheduling decisions, shared by real dispatch and CPU recovery tests."""
def next_action(readiness, service_active, endpoint_valid, checkpoint9000_valid):
    if not readiness: return 'blocked_identity'
    if service_active: return 'wait_existing'
    if endpoint_valid: return 'evaluate_endpoint'
    if checkpoint9000_valid: return 'resume_9000'
    return 'start_parent'

def fixed_tasks(protocol):
    tasks = protocol['task_plan']
    assert [(t['repeat'],t['arm']) for t in tasks] == [
        ('1','J_joint'),('1','A_sh'),('1','S_cov'),
        ('2','S_cov'),('2','A_sh'),('2','J_joint')]
    assert all(t['start']==6000 and t['stop']==12000 for t in tasks)
    return tasks
