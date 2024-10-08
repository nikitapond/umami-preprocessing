import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_dd
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from ftag.hdf5 import H5Writer, H5Reader
import dataclasses
from pathlib import Path
import sys
from typing import Callable, Union
from puma import Histogram, HistogramPlot
import tqdm
import argparse
from upp.classes.reweighting_config import Reweight, ReweightConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()

def join_structured_arrays(arrays: list):
    """Join a list of structured numpy arrays.

    See https://github.com/numpy/numpy/issues/7811

    Parameters
    ----------
    arrays : list
        List of structured numpy arrays to join

    Returns
    -------
    np.array
        A merged structured array
    """
    dtype: list = sum((a.dtype.descr for a in arrays), [])
    newrecarray = np.empty(arrays[0].shape, dtype=dtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]

    return newrecarray

def bin_jets(array: dict, bins: list) -> np.ndarray:
    """Create the histogram and bins for the given resampling variables.

    Parameters
    ----------
    array : dict
        Dict with the loaded jets and the resampling
        variables.
    bins : list
        Flat list with the bins which are to be used.

    Returns
    -------
    hist : np.ndarray, shape(nx1, nx2, nx3,...)
        The values of the selected statistic in each two-dimensional bin.
    out_bins : (N,) array of ints or (D,N) ndarray of ints
        This assigns to each element of `sample` an integer that represents the
        bin in which this observation falls.  The representation depends on the
        `expand_binnumbers` argument.  See `Notes` for details.
    """
    hist, _, out_bins = binned_statistic_dd(
        sample=s2u(array),
        values=None,
        statistic="count",
        bins=bins,
        expand_binnumbers=True,
    )
    out_bins -= 1
    return hist, out_bins



def calculate_weights(
    input_file : str,
    reweights : list[Reweight],
):
    '''
    Generates all the calculate_weights for the reweighting and returns them in a dict
    of the form:
    {
        'group_name' : {
            'repr(reweight)' : {
                'bins': np.ndarray, # The bins used for the histogram
                'histograms' : {
                        label_0 : hists_for_label_0, # np.ndarray
                        label_1 : hists_for_label_1,
                        ...
                    }
                }
        
    }
    
    '''

    print(f"Calculating weights for {len(reweights)} reweights")

    reader = H5Reader(input_file)
    all_vars = {}
    existing_vars = {}
    with h5py.File(input_file, 'r') as f:
        for group in f.keys():
            existing_vars[group] = list(f[group].dtype.names)
        
    # Get the variables we need to reweight
    for rw in reweights:
        rw_group = rw.group
        if rw_group not in all_vars:
            all_vars[rw_group] = []
        if rw.class_var is not None:
            all_vars[rw_group].append(rw.class_var)
        all_vars[rw_group].extend(rw.reweight_vars)
        if 'valid' in existing_vars[rw_group]:
            all_vars[rw_group] += ['valid']
    if "jets" not in all_vars:
        all_vars["jets"] = ["pt"]
    all_vars = {k: list(set(v)) for k,v in all_vars.items()}
    num_in_hists = {}
    all_histograms = {}

    for batch in tqdm.tqdm(reader.stream(all_vars), total=reader.num_jets / reader.batch_size):

        # Keep track of how many items we've used to generate our histograms
        for k, v in batch.items():
            if k not in num_in_hists:
                num_in_hists[k] = v.shape[0]
            else:
                num_in_hists[k] += v.shape[0]

        for rw in reweights:
            rw_group = rw.group
            if rw_group not in batch:
                continue
            data = batch[rw_group]

            if len(data.shape) != 1:
                assert 'valid' in data.dtype.names
                data = data[data['valid']]
            
            if rw.class_var is not None:
                classes = np.unique(data[rw.class_var])
            else:
                classes = [None]
            for cls in classes:
                mask = data[rw.class_var] == cls
                hist, outbins = bin_jets(data[mask][rw.reweight_vars], rw.flat_bins)
                if rw.class_var is not None:
                    cls = str(cls)
                if rw_group not in all_histograms:
                    all_histograms[rw_group] = {}
                if repr(rw) not in all_histograms[rw_group]:
                    all_histograms[rw_group][repr(rw)] = {
                        'bins' : rw.flat_bins,
                        'histograms' : {}
                    }
                if cls not in all_histograms[rw_group][repr(rw)]['histograms']:
                    all_histograms[rw_group][repr(rw)]['histograms'][cls] = hist.copy()
                else:
                    all_histograms[rw_group][repr(rw)]['histograms'][cls] += hist


    # Define for each RW what the target histogram should be. This is either a single 
    # flavour, mean of multiple/all flavours
    all_targets = {}
    for rw in reweights:
        rw_group = rw.group
        if rw_group not in all_histograms:
            raise ValueError(f"Group {rw_group} not found in histograms... What?")
            
        if rw_group not in all_targets:
            all_targets[rw_group] = {}
        
        rw_rep = repr(rw)

        target = None

        if isinstance(rw.class_target, int): 
            target = all_histograms[rw_group][rw_rep]['histograms'][str(rw.class_target)]
        elif isinstance(rw.class_target, str) and rw.class_target == 'mean':
            for cls, hist in all_histograms[rw_group][rw_rep]['histograms'].items():
                
                if target is None:
                    target = hist.copy()
                else:
                    target += hist
            target /= len(all_histograms[rw_group][rw_rep]['histograms'])
        elif isinstance(rw.class_target, str) and rw.class_target == 'min':
            for cls, hist in all_histograms[rw_group][rw_rep]['histograms'].items():
                if target is None:
                    target = hist.copy()
                else:
                    target = np.minimum(target, hist)
        elif isinstance(rw.class_target, str) and rw.class_target == 'max':
            for cls, hist in all_histograms[rw_group][rw_rep]['histograms'].items():
                if target is None:
                    target = hist.copy()
                else:
                    target = np.maximum(target, hist)
        elif isinstance(rw.class_target, str) and rw.class_target == 'uniform':
            target = np.ones_like(all_histograms[rw_group][rw_rep]['histograms'][str(0)])
        elif isinstance(rw.class_target, (list, tuple)):
            for cls, hist in all_histograms[rw_group][rw_rep]['histograms'].items():
                cast_cls_target = tuple(map(str, rw.class_target))
                if cls in cast_cls_target:
                    if target is None:
                        target = hist.copy()
                    else:
                        target += hist
            target /= len(rw.class_target)
        else:
            raise ValueError("Unknown class_target type")
        
        if np.any(target == 0):
            num_zeros = np.sum(target == 0)
            print(f"Target histogram has {num_zeros} bins with zero entries out of total {target.shape} : {repr(rw)}")
        if np.any(target < 0):
            raise ValueError(f"Target histogram has bins with negative entries : {repr(rw)}")
        if np.any(np.isnan(target)):
            raise ValueError(f"Target histogram has bins with NaN entries : {repr(rw)}")

        # Apply the target histogram function
        if rw.target_hist_func is not None:
            target = rw.target_hist_func(target)
        
        

        all_targets[rw_group][rw_rep] = target

    output_weights = {}
    for rw in reweights:
        rw_group = rw.group
        rw_rep = repr(rw)
        if rw_group not in output_weights:
            output_weights[rw_group] = {}
        if rw_rep not in output_weights[rw_group]:
            output_weights[rw_group][rw_rep] = {}
        output_weights[rw_group][rw_rep] = {
            'weights' : {},
            'bins' : all_histograms[rw_group][rw_rep]['bins'],
            'rw_vars' : rw.reweight_vars,
            'class_var' : rw.class_var,
        }
        idx_below_min = None
        for cls, hist in all_histograms[rw_group][rw_rep]['histograms'].items():

            this_idx_below_min = (hist == 0) #| (all_targets[rw_group][rw_rep] == 0)
            output_weights[rw_group][rw_rep]['weights'][cls] = np.where(hist > 0, all_targets[rw_group][rw_rep] / hist, 0)
            if idx_below_min is None:
                idx_below_min = this_idx_below_min
            else:
                idx_below_min |= this_idx_below_min
        # if np.any(idx_below_min):
        #     for cls, hist in all_histograms[rw_group][rw_rep]['histograms'].items():
        #         output_weights[rw_group][rw_rep]['weights'][cls][idx_below_min] = 0
            
    return output_weights

def save_weights_hdf5(weights_dict, filename):
    '''
    Saves the weights to an HDF5 file, with the following structure:
    {
        'group_name' : {
            'repr(reweight)' : {
                'bins': np.ndarray, # The bins used for the histogram
                'histograms' : {
                        label_0 : hists_for_label_0, # np.ndarray
                        label_1 : hists_for_label_1,
                        ...
                    }
                }
            ...
    }
    Such that they can be loaded by `load_weights_hdf5`
    '''
    filename.parent.mkdir(exist_ok=True, parents=True)
    with h5py.File(filename, 'w') as f:
        for group, data in weights_dict.items():
            group_obj = f.create_group(group)
            for reweight_name, reweight_data in data.items():
                reweight_group = group_obj.create_group(reweight_name)
                
                # Create a group for bins, as it's a list of arrays
                bins_group = reweight_group.create_group('bins')
                for i, bin_array in enumerate(reweight_data['bins']):
                    bins_group.create_dataset(f'bin_{i}', data=bin_array)

                reweight_group.create_dataset('rw_vars', data=np.array(reweight_data['rw_vars'], dtype=h5py.special_dtype(vlen=str)))
                reweight_group.create_dataset('class_var', data=np.array([reweight_data['class_var']], dtype=h5py.special_dtype(vlen=str)))

                # Save histograms
                hist_group = reweight_group.create_group('weights')
                for label, hist in reweight_data['weights'].items():
                    hist_group.create_dataset(f'{label}', data=hist)
                

def load_weights_hdf5(filename):
    '''
    Loads the weights from an HDF5 file, see `save_weights_hdf5` for the structure
    '''    
    weights_dict = {}
    with h5py.File(filename, 'r') as f:
        # Iterate through the groups in the file (top-level groups represent 'group_name')
        for group in f.keys():
            weights_dict[group] = {}
            group_obj = f[group]
            # For each group, iterate through the reweight names
            for reweight_name in group_obj.keys():
                reweight_group = group_obj[reweight_name]
                
                # Load the bins, which is now a list of arrays
                bins_group = reweight_group['bins']
                bins = [bins_group[f'bin_{i}'][:] for i in range(len(bins_group))]
                
                reweight_vars = [var.decode('utf-8') for var in reweight_group['rw_vars'][:]]
                class_var = [var.decode('utf-8') for var in reweight_group['class_var'][:]][0]
                # Load the histograms
                histograms = {}
                hist_group = reweight_group['weights']
                for label in hist_group.keys():
                    histograms[label] = hist_group[label][:]
                
                # Reconstruct the structure
                weights_dict[group][reweight_name] = {
                    'bins': bins,
                    'weights': histograms,
                    'rw_vars': reweight_vars,
                    'class_var': class_var,
                }
    return weights_dict



def get_sample_weights(batch, calculated_weights, scale : dict):
    '''
    Parameters
    ----------
    batch : dict
        A dictionary of numpy arrays, where the keys are the group names
        and the values are the structured arrays of the data
    calculated_weights : dict
        A dictionary of the calculated weights, as returned by `calculate_weights`
    scale : dict
        A dictionary of the scaling factors for the weights, to ensure that the 
        sum of all weights is equal to the number of jets. If this is empty,
        it will be populated with the scaling factors
    '''
    sample_weights = {}
    for group, reweights in calculated_weights.items():
        if group not in sample_weights:
            sample_weights[group] = {}
        if group not in scale:
            scale[group] = {}
        is_1d = len(batch[group].shape) == 1
        if is_1d:
            valid_indices = None
            to_dump = batch[group]
        else:
            valid_indices = np.nonzero(batch[group]['valid'])
            to_dump = batch[group][batch[group]['valid']]
        
        for rwkey, rw in reweights.items():
            
            rw_vars = rw['rw_vars']
            class_var = rw['class_var']
            
            _, bins = bin_jets(to_dump[rw_vars], rw['bins'])
            # Enforce that bins are of shape (nvars, num_objects)
            if len(rw['bins']) == 1:
                bins = np.expand_dims(bins, axis=0)

            this_weights = np.zeros(to_dump[class_var].shape, dtype=float)
            try:
                # This is SUPER slow - as we iterate each object, but its 
                for i in range(this_weights.shape[0]):
                    
                    bin_idx = bins[:, i]
                    cls = to_dump[class_var][i]
                    thishist = rw['weights'][str(cls)][tuple(bin_idx)]
                    this_weights[i] = thishist
            except Exception as e:
                print(f"Error in {group} {rwkey}")
                raise
            
            if rwkey not in scale[group]:
                
                # We return the scale such that the sum of all weights is equal 
                # to the number of jets
                scale[group][rwkey] = 1/np.mean(this_weights)
            if valid_indices is not None:
                weights_out = np.zeros(batch[group].shape, dtype=float)
                weights_out[valid_indices] = this_weights * scale[group][rwkey]
                sample_weights[group][rwkey] = weights_out
            else:
                sample_weights[group][rwkey] = this_weights * scale[group][rwkey]
        
    sample_w_as_struct_arr = {}

    for group, reweights in sample_weights.items():
        dtype = [(key, arr.dtype) for key, arr in reweights.items()]

        structured_array = np.zeros(next(iter(reweights.values())).shape, dtype=dtype)
        for key in reweights:
            structured_array[key] = reweights[key]
        sample_w_as_struct_arr[group] = structured_array
        
    return sample_w_as_struct_arr, scale

def write_sample_with_weights(
    input_file : str,
    output_file : str,
    weights : dict,
):
    print("Writing weights to ", output_file)
    all_groups = {}
    with h5py.File(input_file, 'r') as f:
        for group in f.keys():
            all_groups[group] = None
    reader = H5Reader(input_file)
    num_jets = reader.num_jets
    writer = None
    additional_vars = {}

    for group, reweight in weights.items():
        
        additional_vars[group] = list(reweight.keys())
        
    dtypes = { k : v.descr for k, v in reader.dtypes().items() }

    for group, rw_output_names in additional_vars.items():
        for rw_name in rw_output_names:
            dtypes[group] += [(rw_name, 'f4')]
    for group in dtypes:
        dtypes[group] = np.dtype(dtypes[group])
    
    # The amount we scale all final weights by, to ensure that the sum of all weights 
    # is (approximatly) equal to the number of jets
    scale = {}
    for batch in tqdm.tqdm(reader.stream(all_groups, num_jets=num_jets), total=num_jets / reader.batch_size):
        all_sample_weights, scale = get_sample_weights(batch, weights, scale)
        to_write = {}
        for key in batch.keys():
            if key in all_sample_weights:
                to_write[key] = join_structured_arrays([batch[key], all_sample_weights[key]])
            else:
                to_write[key] = batch[key]
        # print(batch.keys())
        if writer is None:
            shapes = {k: (num_jets,) + v.shape[1:] for k, v in to_write.items()}
            # print(shapes)
            writer = H5Writer(output_file, dtypes, shapes, shuffle=False,)
            writer.copy_attrs(input_file)
        writer.write(to_write)
 


def plot_hist(data, var, bins, weights, fname, label='flavour_label'):

    plot = HistogramPlot(
        xlabel=f"{var}",
        ylabel="Normalised Number of objects",
        logy=True,
        bins=bins,
        n_ratio_panels=1,

    )

    has_ratio = False
    for i, cls in enumerate(np.unique(data[label])):
        mask = ((data[label] == cls) & (~np.isnan(data[var])))
        if np.sum(mask) == 0:
            continue
        plot.add(
            Histogram(data[mask][var], label=f"Class {cls}"), 
            reference=True if not has_ratio else False,
        )
        has_ratio = True

    path = plot_dir / f"{fname}_unweighted_{var}.png"
    plot.draw()
    plot.savefig(path)
    print('\t', path)
    
    plot = HistogramPlot(
        xlabel=f"{var}",
        ylabel="Normalised Number of objects",
        logy=True,
        bins=bins,
        n_ratio_panels=1,

    )

    has_ratio = False
    for i, cls in enumerate(np.unique(data[label])):
        mask = ((data[label] == cls) & (~np.isnan(data[var])))
        if np.sum(mask) == 0:
            continue
        plot.add(
            Histogram(data[mask][var], label=f"Class {cls}", weights=weights[mask]), 
            reference=True if not has_ratio else False,
        )
        has_ratio = True
    path = plot_dir / f"{fname}_reweighted_{var}.png"
    plot.draw()
    plot.savefig(path)

    print('\t', path)

    w_bins = np.logspace(np.log10(np.min(weights[weights > 0])), np.log10(np.max(weights)), 100)
    plot = HistogramPlot(
        xlabel="Weights",
        ylabel="Normalised Number of objects",
        logy=True,
        bins=w_bins,
        n_ratio_panels=1,
    )

    has_ratio = False
    for i, cls in enumerate(np.unique(data[label])):
        mask = ((data[label] == cls) & (~np.isnan(data[var])))
        if np.sum(mask) == 0:
            continue
        plot.add(
            Histogram(weights[mask], label=f"Class {cls}"), 
            reference=True if not has_ratio else False,
        )
        has_ratio = True
    path = plot_dir / f"{fname}_weights.png"
    plot.draw()
    plot.savefig(path)

def make_plots(all_rw, fpath):

    h5 = h5py.File(fpath, "r")

    for rw in all_rw:
        data = h5[rw.group][:]

        for v, b in zip(rw.reweight_vars, rw.bins):
            weights = h5[rw.group][f'{repr(rw)}']
            plot_hist(data, v, b, weights, rw, label=rw.class_var)




# input_file="/home/xzcappon/phd/datasets/vertexing_120m/output/pp_output_val.h5"
# out_file="/home/xzcappon/phd/datasets/vertexing_120m/output/pp_output_val_weighted_test.h5"
# hists_dir="/home/xzcappon/phd/datasets/vertexing_120m/output/hists_val_test.h5"
# plot_dir="rw_plots/val"
# plot_dir = Path(hists_dir).parent / plot_dir
# plot_dir.mkdir(exist_ok=True, parents=True)

# file = h5py.File(input_file, "r")

# pt_bins = np.linspace(20_000, 6_000_000, 50)
# abs_eta_bins = np.linspace(0, 2.5, 20)
# eta_bins = np.linspace(-2.5, 2.5, 40)
# th_pt_bins = np.linspace(5_000, 5600000.0, 50)
# th_lxy_bins = np.concatenate([
#     np.linspace(0, 50, 50)[:-1],
#     np.linspace(50, 250, 20)[:-1],
#     np.linspace(250, 1500, 20),
#     np.array([3000])
#     ])

# phi_bins = np.linspace(-3.2, 3.2, 50)

# all_reweights = [
    
#     Reweight(
#         group='truth_hadrons',
#         reweight_vars=['pt'],
#         bins=[th_pt_bins],
#         class_var='flavour',
#         class_target='mean',
#     ),
#     Reweight(
#         group='truth_hadrons',
#         reweight_vars=['eta'],
#         bins=[eta_bins],
#         class_var='flavour',
#         class_target='mean',
#     ),
#     Reweight(
#         group='truth_hadrons',
#         reweight_vars=['phi'],
#         bins=[phi_bins],
#         class_var='flavour',
#         class_target='mean',
#     ),
#     Reweight(
#         group='truth_hadrons',
#         reweight_vars=['energy'],
#         bins=[pt_bins],
#         class_var='flavour',
#         class_target='mean',
#     ),
#     Reweight(
#         group='truth_hadrons',
#         reweight_vars=['Lxy'],
#         bins=[th_lxy_bins],
#         class_var='flavour',
#         class_target='mean',
#     ),
#     Reweight(
#         group='jets',
#         reweight_vars=['pt_btagJes', 'absEta_btagJes'],
#         bins=[pt_bins, abs_eta_bins],
#         class_var='flavour_label',
#         class_target='mean',
#     ),
# ]




def main(args=None):
    if args is None:
        args = parse_args()

    config = ReweightConfig.from_file(args.config)

    if not config.hist_path.exists():
        calculated_weights = calculate_weights(config.train_in_path, config.reweights)
        save_weights_hdf5(calculated_weights, config.hist_path)
    else:
        print("Loading weights from ", config.hist_path)

    calculated_weights = load_weights_hdf5(config.hist_path)

    print("Running training file...")
    write_sample_with_weights(
        config.train_in_path,
        config.train_out_path,
        calculated_weights
    )

    make_plots(config.reweights, config.plot_dir / 'training')

    print("Running validation file...")
    write_sample_with_weights(
        config.val_in_path,
        config.val_out_path,
        calculated_weights
    )
    make_plots(config.reweights, config.plot_dir / 'validation') 

if __name__ == "__main__":
    main()