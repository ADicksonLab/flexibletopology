from flexibletopology.utils.utils import save_anigsg_model



MODEL_SAVE_PATH = 'inputs/anigsg_model.pt'

if __name__=="__main__":

    max_wavelet_scale = 4
    radial_cutoff = 2.0
    sm_operators = (True, True, False)
    save_path = MODEL_SAVE_PATH
    save_anigsg_model(max_wavelet_scale,
                   radial_cutoff,
                   sm_operators,
                   MODEL_SAVE_PATH)
