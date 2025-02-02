import bundle_submissions
import dnd_denoise
import pytorch_wrapper
import sys
sys.path.append("/please/replace/with/path/to/dnd/submission/kit/py/")


data_folder = "/please/replace/with/your/dnd/data/folder/"


def eval_dnd(experiment, mode, out_folder, use_cuda=True, preproc=None, postproc=None):
    net = experiment.net
    net.eval()

    def fun(noisy, nlf):
        if preproc:
            noisy = preproc(noisy, nlf)
        denoised = net(noisy)
        if postproc:
            denoised = postproc(denoised, nlf)
        return denoised.data
    wrapper_fun = pytorch_wrapper.pytorch_denoiser(fun, use_cuda=use_cuda)
    if mode == "raw":
        dnd_denoise.denoise_raw(wrapper_fun, data_folder, out_folder)
        bundle_submissions.bundle_submissions_raw(out_folder)
    elif mode == "raw_full":
        dnd_denoise.denoise_raw_full(wrapper_fun, data_folder, out_folder)
        bundle_submissions.bundle_submissions_raw(out_folder)
    else:
        dnd_denoise.denoise_srgb(wrapper_fun, data_folder, out_folder)
        bundle_submissions.bundle_submissions_srgb(out_folder)
