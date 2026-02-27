import os
import zipfile
from glob import glob
from omegaconf import OmegaConf


def load_conf(config_path):
    base_conf = OmegaConf.load(config_path)
    config_dir = '/'.join(config_path.split('/')[:-1])
    confs = [
        OmegaConf.load(os.path.join(config_dir, path))
        for path in base_conf.configs
    ]
    return OmegaConf.merge(base_conf, *confs)


def make_archive(out_name, root_dir, files) -> None:
    if not root_dir.endswith('/'):
        root_dir = root_dir+'/'
    with zipfile.ZipFile(out_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            assert file.startswith(root_dir)
            full_path = os.path.join(root_dir, file)
            zipf.write(full_path, arcname=file[len(root_dir):])


def prepare_folders(conf):
    tb_path = os.path.join(conf.logger.tensorboard.root_dir, conf.common.experiment_name)
    os.makedirs(conf.checkpoint.root_dir, exist_ok=True)
    os.makedirs(tb_path, exist_ok=True)
    version = max([0] + [int(i.split('_')[-1]) for i in os.listdir(tb_path) if i.startswith('version_')])
    log_dir = os.path.join(conf.checkpoint.root_dir, f'version_{version}')
    os.makedirs(log_dir, exist_ok=True)

    conf.common.conf_path = os.path.join(conf.checkpoint.root_dir, "config.yaml")
    OmegaConf.save(conf, conf.common.conf_path)
    OmegaConf.save(conf, os.path.join(log_dir, "config.yaml"))

    archive_name = os.path.join(log_dir, "config.yaml")
    code_path = '/'.join(os.path.abspath(__file__ ).split('/')[:-2])
    make_archive(archive_name, code_path, glob(os.path.join(code_path,'**/*.py'), recursive=True))