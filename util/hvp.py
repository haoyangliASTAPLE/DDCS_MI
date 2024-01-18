""" Use scipy/ARPACK implicitly restarted lanczos to find top k eigenthings
This code solve generalized eigenvalue problem for operators or matrices
Format adapted from lanczos.py in hessian-eigenthings
"""
import torch
import numpy as np
import torch.nn.functional as F
import torch.autograd
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import eigsh
from warnings import warn

def lanczos(
    operator,
    num_eigenthings=10,
    which="LM",
    max_steps=20,
    tol=1e-6,
    num_lanczos_vectors=None,
    # init_vec=None,
    use_gpu=False,
):
    """
    https://github.com/noahgolmant/pytorch-hessian-eigenthings/blob/8ff8b3907f2383fe1fdaa232736c8fef295d8131/hessian_eigenthings/lanczos.py#L11
    Use the scipy.sparse.linalg.eigsh hook to the ARPACK lanczos algorithm
    to find the top k eigenvalues/eigenvectors.
    Parameters
    -------------
    operator: power_iter.Operator
        linear operator to solve.
    num_eigenthings : int
        number of eigenvalue/eigenvector pairs to compute
    which : str ['LM', SM', 'LA', SA']
        L,S = largest, smallest. M, A = in magnitude, algebriac
        SM = smallest in magnitude. LA = largest algebraic.
    max_steps : int
        maximum number of arnoldi updates
    tol : float
        relative accuracy of eigenvalues / stopping criterion
    num_lanczos_vectors : int
        number of lanczos vectors to compute. if None, > 2*num_eigenthings
    init_vec: [torch.Tensor, torch.cuda.Tensor]
        if None, use random tensor. this is the init vec for arnoldi updates.
    use_gpu: bool
        if true, use cuda tensors.
    Returns
    ----------------
    eigenvalues : np.ndarray
        array containing `num_eigenthings` eigenvalues of the operator
    eigenvectors : np.ndarray
        array containing `num_eigenthings` eigenvectors of the operator
    """
    if isinstance(operator.size, int):
        size = operator.size
    else:
        size = operator.size[0]
    shape = (size, size)

    if num_lanczos_vectors is None:
        num_lanczos_vectors = min(2 * num_eigenthings, size - 1)
    if num_lanczos_vectors < 2 * num_eigenthings:
        warn(
            "[lanczos] number of lanczos vectors should usually be > 2*num_eigenthings"
        )

    def _scipy_apply(x):
        x = torch.from_numpy(x)
        if use_gpu:
            x = x.cuda()
        out = operator.apply(x.float())
        out = out.cpu().numpy()
        return out

    scipy_op = ScipyLinearOperator(shape, _scipy_apply)
    # if init_vec is None:
    #     init_vec = np.random.rand(size)
    # elif isinstance(init_vec, torch.Tensor):
    #     init_vec = init_vec.cpu().numpy()

    eigenvals, eigenvecs = eigsh(
        A=scipy_op,
        k=num_eigenthings,
        which=which,
        maxiter=max_steps,
        tol=tol,
        ncv=num_lanczos_vectors,
        return_eigenvectors=True,
    )
    return eigenvals, eigenvecs.T

class Operator:
    """
    maps x -> Lx for a linear operator L
    https://github.com/noahgolmant/pytorch-hessian-eigenthings/blob/8ff8b3907f2383fe1fdaa232736c8fef295d8131/hessian_eigenthings/operator.py#L3
    """
    def __init__(self, size):
        self.size = size

    def apply(self, vec):
        """
        Function mapping vec -> L vec where L is a linear operator
        """
        raise NotImplementedError

class GANHVPOperator(Operator):
    """ Uses backward autodifferencing to compute HVP for unary and binary D """
    def __init__(
            self,
            model,
            code,
            criterion,
            use_gpu=True,
            preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True),
    ):
        if use_gpu:
            device = "cuda"
            self.device = device
        if hasattr(model,"parameters"):
            for param in model.parameters():
                param.requires_grad_(False)
        if hasattr(criterion,"parameters"):
            for param in criterion.parameters():
                param.requires_grad_(False)

        self.model = model  # model is the generator 
        self.preprocess = preprocess  # preprocess the generated `x` (e.g. images) before measuring dissimilarity. 
        self.criterion = criterion  # distance function in generated space `d(x_1,x_2)`. or objective `d(x_1)` in activation case. 
        self.code = code.clone().requires_grad_(False).float().to(device)  # reference code z_0 to measure Hessian at. 
        self.size = 3 * 256 * 256

        # self.perturb_vec = 0.0001 * torch.randn((1, self.size), dtype=torch.float32).requires_grad_(True).to(
        #     device)  # dimension debugged @Sep 10
        self.perturb_vec = 0.0001 * torch.randn((1, 512), dtype=torch.float32).cuda()

        with torch.no_grad():
            self.img_ref = self.model(self.code, None, noise_mode='const')  # forward the feature vector through the GAN
            self.img_pertb = self.model(self.code + self.perturb_vec, None, noise_mode='const') - self.img_ref
        self.img_pertb.requires_grad_(True)

        d_sim = self.criterion(self.preprocess(self.img_ref), self.preprocess(self.img_ref+self.img_pertb))
        # similarity metric between 2 images.

        gradient = torch.autograd.grad(d_sim, self.img_pertb, create_graph=True, retain_graph=True)[0]
        # 1st order gradient, saved, enable 2nd order gradient. 

        self.gradient = gradient.view(-1)

    '''
    def select_code(self, code):
        """ Select a (new) reference code `z` to compute Hessian at. H|_z
        Input: 
            code: torch tensor of shape `[1, n]` 
        """
        self.code = code.clone().requires_grad_(False).float().to(self.device) # torch.float32
        self.size = self.code.numel()
        self.perturb_vec = torch.zeros((1, self.size), dtype=torch.float32).requires_grad_(True).to(self.device)
        self.img_ref = self.model(self.code, None)  # forward the feature vector through the GAN: G(z), without gradient.
        img_pertb = self.model(self.code + self.perturb_vec, None)  # forward feature vector + perturb: G(z+dz), enable gradient. 
        d_sim = self.criterion(self.preprocess(self.img_ref), self.preprocess(img_pertb))  # compute d = D(G(z), G(z+dz)), enable gradient
        gradient = torch.autograd.grad(d_sim, self.perturb_vec, create_graph=True, retain_graph=True)[0] # compute `\partial d/\partial dz` 
        self.gradient = gradient.view(-1)
        self.size = self.perturb_vec.numel()
    '''

    def apply(self, vec):
        """ Compute Hessian Vector Product(HVP) of `vec` using forward differencing method. 
        Here we implement the forward difference approximation of HVP.
              $Hv|_x = \partial_dz g(x)^Tv|_{x+dz}$
        Input:
            vec: Vector to product with Hessian. a torch vector of shape `[1, n]`

        Returns:
            hessian_vec_prod: H*vec where H is the hessian of the output of D, w.r.t. latent input to G.
        """
        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(
            self.gradient, self.img_pertb, grad_outputs=vec, only_inputs=True, retain_graph=True
        )
        hessian_vec_prod = grad_grad[0].view(-1)  # torch.cat([g.view(-1) for g in grad_grad]) #.contiguous()
        return hessian_vec_prod

    def vHv_form(self, vec):
        """ Compute Bilinear form of Hessian with `vec` using Hessian Vector Product method
        Input:
            vec: Vector to compute vHv. a torch vector of shape `[1, n]`

        Returns:
            vhv: a torch scalar. Bilinear form vec.T*H*vec. where H is the hessian of D output.
        """
        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(
            self.gradient, self.img_pertb, grad_outputs=vec, only_inputs=True, retain_graph=True
        )
        hessian_vec_prod = grad_grad[0].view(-1)
        vhv = (hessian_vec_prod * vec).sum()
        return vhv

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in [self.img_pertb]:
            if p.grad is not None:
                p.grad.data.zero_()
