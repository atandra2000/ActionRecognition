import os
import tempfile
import pytest
import yaml
from src.utils.config import load_config, Config


class TestLoadConfig:
    def test_parses_yaml_correctly(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'model': {'num_classes': 60, 'num_keypoints': 17},
                'training': {'num_epochs': 50},
                'project_name': 'test_project'
            }, f)
            f.flush()
            config = load_config(f.name)
            os.unlink(f.name)

        assert config.model.num_classes == 60
        assert config.model.num_keypoints == 17
        assert config.training.num_epochs == 50
        assert config.project_name == 'test_project'

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path/config.yaml')

    def test_defaults_preserved(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'model': {'num_classes': 10}}, f)
            f.flush()
            config = load_config(f.name)
            os.unlink(f.name)

        assert config.model.num_classes == 10
        assert config.model.num_keypoints == 25
        assert config.training.num_epochs == 120
