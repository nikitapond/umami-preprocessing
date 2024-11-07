from dataclasses import dataclass
from typing import Callable, Union
import numpy as np
from pathlib import Path
import yaml
from yamlinclude import YamlIncludeConstructor
# support inclusion of yaml files in the config dir
YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.SafeLoader, base_dir=Path(__file__).parent.parent / "configs"
)

@dataclass
class Reweight:
    group : str # The group our variables in the h5 file are in
    reweight_vars : list[str] # The variables we want to reweight
    bins : list[np.ndarray] # The bins we want to use for the reweighting
    class_var : str # The variable which contains the label we resample over, e.g. flavour
    class_target : int | tuple | str = None
    
    target_hist_func: Union[Callable, None] = None
    target_hist_func_name: str | None = None

    # TODO - this is the same as in resampling, maybe can cleanup
    def get_bins_x(self, bins_x, upscale=1):
        flat_bins = []
        for i, sub_bins_x in enumerate(bins_x):
            start, stop, nbins = sub_bins_x
            b = np.linspace(start, stop, nbins * upscale + 1)
            if i > 0:
                b = b[1:]
            flat_bins.append(b)
        return np.concatenate(flat_bins)

    @property
    def flat_bins(self):
        return [self.get_bins_x(self.bins[k]) for k in self.reweight_vars]

    def __post_init__(self):
        if isinstance(self.class_target, str):
            # TODO also target mean etc?
            if self.class_target not in ['mean', 'min', 'max', 'uniform']:
                raise ValueError("class_target must be either 'mean', 'min', 'max' or an integer")
            
        if self.target_hist_func is not None:    
            if self.target_hist_func_name is None:
                self.target_hist_func_name = self.target_hist_func.__name__
    
    def __repr__(self):
        target_str = 'target_' 
        if self.target_hist_func_name is not None:
            target_str += f"{self.target_hist_func_name}_"
        if self.class_target is not None:
            if isinstance(self.class_target, (list, tuple)):
                target_str += '_'.join(map(str, self.class_target))
            else:
                target_str += f"{self.class_target}_{self.class_var}"
        else:
            target_str += 'none'
        return f"weight_{self.group}_{'_'.join(self.reweight_vars)}_{target_str}"

@dataclass
class ReweightConfig:

    reweights : list[Reweight]
    hist_path : Path
    output_dir : Path
    plot_dir : Path | None= None

    def __post_init__(self):
        self.reweights = [Reweight(**reweight) for reweight in self.reweights]
        self.hist_path = Path(self.hist_path)
        self.output_dir = Path(self.output_dir)
        self.train_in_path = self.output_dir / "pp_output_train.h5"
        self.train_out_path = self.output_dir / "pp_output_train_reweighted.h5"
        self.val_in_path = self.output_dir / "pp_output_val.h5"
        self.val_out_path = self.output_dir / "pp_output_val_reweighted.h5"

    @classmethod
    def from_file(cls, config_path: Path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        reweighting_config = config.get("reweighting")
        
        if reweighting_config is None:
            raise ValueError("No reweighting configuration found in config file")
        
        reweights = reweighting_config["reweights"]

        if not (hist_path := reweighting_config.get("hist_path")):
            hist_path = Path(config["global"]["base_dir"]) / "hists/reweighting.h5"
            print("No hist_dir specified in config file, using default path: ", hist_path)
        
        output_dir = Path(config["global"]["base_dir"]) / config["global"].get("out_dir", "output")

        plot_dir = config.get("plot_dir")
        if plot_dir is not None:
            plot_dir = Path(plot_dir)
        else:
            plot_dir = output_dir / "reweight_plots"

        return cls(
            reweights=reweights, 
            hist_path=hist_path, 
            output_dir=output_dir,
            plot_dir=plot_dir)
