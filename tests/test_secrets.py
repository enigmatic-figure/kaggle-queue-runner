import sys
from types import SimpleNamespace
from queue_runner.cli import get_secret


def test_control_token_accepts_notebook_label_without_second_underscore(monkeypatch):
    monkeypatch.delenv('QUEUE_CONTROL_TOKEN', raising=False)
    monkeypatch.setenv('QUEUE_CONTROLTOKEN', 'example-control-token')
    assert get_secret('QUEUE_CONTROL_TOKEN') == 'example-control-token'


def test_control_token_alias_is_read_from_kaggle_secrets(monkeypatch):
    monkeypatch.delenv('QUEUE_CONTROL_TOKEN', raising=False)
    monkeypatch.delenv('QUEUE_CONTROLTOKEN', raising=False)
    class Client:
        def get_secret(self, label):
            if label == 'QUEUE_CONTROLTOKEN':
                return 'example-kaggle-token'
            raise KeyError(label)
    monkeypatch.setitem(sys.modules, 'kaggle_secrets', SimpleNamespace(UserSecretsClient=Client))
    assert get_secret('QUEUE_CONTROL_TOKEN') == 'example-kaggle-token'


def test_canonical_environment_token_takes_precedence(monkeypatch):
    monkeypatch.setenv('QUEUE_CONTROL_TOKEN', 'canonical-example')
    monkeypatch.setenv('QUEUE_CONTROLTOKEN', 'alias-example')
    assert get_secret('QUEUE_CONTROL_TOKEN') == 'canonical-example'
