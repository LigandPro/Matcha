from matcha.utils.log import get_logger

logger = get_logger(__name__)



def construct_output_dict(preds, dataset):
    output_dict = {}
    processed_names = set()
    for complex in dataset.complexes:
        uid_full = complex.name
        if uid_full in processed_names:
            continue
        processed_names.add(uid_full)
        uid_real = uid_full.split('_conf')[0]

        preds_list = preds[uid_full]
        if len(preds_list) == 0:
            continue

        if uid_real not in output_dict:
            output_dict[uid_real] = {
                'sample_metrics': [],
                'orig_mol': complex.ligand.orig_mol,
            }
        samples = []
        for pred in preds_list:
            sample = {
                'pred_pos': pred['transformed_orig'] + pred['full_protein_center'].reshape(1, 3),
                'error_estimate_0': pred.get('error_estimate_0', 0),
            }
            samples.append(sample)
        output_dict[uid_real]['sample_metrics'].extend(samples)
    return output_dict
