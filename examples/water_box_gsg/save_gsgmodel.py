from flexibletopology.utils.utils import save_gsg_model


MODEL_SAVE_PATH = 'inputs/gsg_model.pt'

if __name__ == "__main__":

    max_wavelet_scale = 4
    radial_cutoff = 2.0
    sm_operators = (True, True, False)
    save_path = MODEL_SAVE_PATH
    platform = 'cuda'
    save_gsg_model(max_wavelet_scale,
                   radial_cutoff,
                   sm_operators,
                   platform,
                   MODEL_SAVE_PATH)
