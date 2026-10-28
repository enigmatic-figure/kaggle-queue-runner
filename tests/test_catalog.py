import copy,json
import pytest
from queue_runner import catalog
from queue_runner.contracts import validate


def test_selected_set_and_backlog_are_portable_pinned_json_configuration():
    files={
        'sets.json':{'version':1,'sets':{'two-embedders':{'services':[
            {'id':'e0','profile':'profiles/e0.json','enabled':True},{'id':'e1','profile':'profiles/e1.json','enabled':True}]}},
            'backlogs':{'profiling':'jobs.json'}},
        'selection.json':{'active_set':'two-embedders','backlogs':['profiling'],'idle_mode':'await_work','checkpoint_reserve_seconds':60},
        'jobs.json':{'jobs':[{'id':'profile-v1','script':'profile.py','service_set':'two-embedders'}]}}
    original={'version':1,'jobs':[],'configuration':{'registry':'sets.json','selection':'selection.json'}}
    result=catalog.resolve(original,lambda p:copy.deepcopy(files[p]))
    assert len(result['services'])==2 and result['jobs'][0]['id']=='profile-v1'
    assert result['policy']['idle_mode']=='await_work'
    assert original['jobs']==[]
    validate(result)


def test_catalog_rejects_missing_sets_duplicate_jobs_and_external_paths():
    files={'r.json':{'version':1,'sets':{},'backlogs':{}},'s.json':{'active_set':'missing'}}
    value={'version':1,'jobs':[],'configuration':{'registry':'r.json','selection':'s.json'}}
    with pytest.raises(ValueError):catalog.resolve(value,lambda p:files[p])
    with pytest.raises(ValueError):catalog.resolve({**value,'configuration':{'registry':'../r.json','selection':'s.json'}},lambda p:files[p])


def test_all_published_sets_and_backlogs_materialize_and_validate():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    registry=json.loads((root/'catalog/sets.json').read_text())
    for name in registry['sets']:
        read=lambda p: {'active_set':name,'backlogs':list(registry['backlogs'])} if p=='selection.json' else json.loads((root/p).read_text())
        result=catalog.resolve({'version':1,'jobs':[],'configuration':{'registry':'catalog/sets.json','selection':'selection.json'}},read)
        validate(result)
