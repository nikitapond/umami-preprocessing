import tempfile
from pathlib import Path

from upp.classes.reweighting_config import ReweightConfig
import yaml

class TestReweightingConfig:

    def test_bins(self):

        # dummy_config = {
        #     global:
        #     jets_name: jets
        #     batch_size: 1_000_000
        #     num_jets_estimate: 25_000_000
        #     base_dir: /home/xzcappon/phd/datasets/flavour_tagging/weighting_studies/noresample_v2
        #     ntuple_dir: /home/xzcappon/central_dumps/p6057/gn2v01/full/
        # }
        dummy_config = {
            "global": {
                "jets_name": "jets",
                "batch_size": 1_000_000,
                "num_jets_estimate": 25_000_000,
                "base_dir": "/home/xzcappon/phd/datasets/flavour_tagging/weighting_studies/noresample_v2",
                "ntuple_dir": "/home/xzcappon/central_dumps/p6057/gn2v01/full/"
            },
            "reweighting" : {
                "reweights" : [
                    {
                        "group" : "jets",
                        "reweight_vars" : ["pt_btagJes", "absEta_btagJes"],
                        "class_var" : "flavour",
                        "class_target" : "mean",
                        "bins" : {
                            "pt_btagJes": [[20_000, 250_000, 50], [250_000, 1_000_000, 50], [1_000_000, 6_000_000, 50]],
                            "absEta_btagJes": [[0, 2.5, 20]]
                        }
                    }
                ]
            }
        }
        # write dummy config to temp dir
        # write dummy config to temp dir
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(dummy_config, f)

            reweighting_config = ReweightConfig.from_file(config_file)
            print(reweighting_config.reweights[0].bins)
            print(reweighting_config.reweights[0].flat_bins)