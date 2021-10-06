import numpy as np
from TransformNet import TransformNet,Mapping
from torch import optim
import torch
import geoopt
from scipy.linalg import sqrtm
def transform_SW(src,target,src_label,origin,n):
    np.random.seed(1)
    torch.manual_seed(1)
    sliced_values = np.zeros_like(src)

    for k in range(n):
        u = np.random.randn(3)
        u /= np.linalg.norm(u)
        proj_source = (src.dot(u).astype('long'))
        proj_target = (target.dot(u).astype('long'))

        proj_source_sorted = np.argsort(proj_source)
        proj_target_sorted = np.argsort(proj_target)

        sliced_values[proj_source_sorted] += target[proj_target_sorted]

    sliced_values /= n


    img = sliced_values[src_label].reshape(origin.shape).astype('uint8')
    return img
def transform_maxSW(src,target,src_label,origin):
    np.random.seed(1)
    torch.manual_seed(1)
    source_values_cuda = torch.from_numpy(src).float().cuda()
    target_values_cuda = torch.from_numpy(target).float().cuda()
    size=3
    manifold = geoopt.manifolds.Sphere()
    mu = geoopt.ManifoldParameter(manifold.random(1, size).cuda(), manifold=manifold,requires_grad=True).cuda()
    op = geoopt.optim.RiemannianAdam([mu], lr=0.01)
    posterior_samples_d = source_values_cuda.detach()
    # posterior_samples_d= posterior_samples_d/torch.max(posterior_samples_d,dim=1,keepdim=True)[0]
    prior_samples_d = target_values_cuda.detach()
    # prior_samples_d= prior_samples_d/torch.max(prior_samples_d,dim=1,keepdim=True)[0]
    for _ in range(10):

        first_projections = posterior_samples_d.matmul(mu.transpose(0, 1))

        second_projections = prior_samples_d.matmul(mu.transpose(0, 1))
        wasserstein_distance = torch.abs((torch.sort(first_projections.transpose(0, 1), dim=1)[0] -
                                          torch.sort(second_projections.transpose(0, 1), dim=1)[0]))
        wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance,2), dim=1), 1. / 2)
        wasserstein_distance= torch.pow(torch.pow(wasserstein_distance, 2).mean(),1./2)
        loss = -wasserstein_distance

        op.zero_grad()
        loss.backward()
        op.step()
    sliced_values = np.zeros_like(src)
    projections = mu.detach().cpu().numpy()
    for k in range(1):
        u = projections[k]
        # print(u.shape)
        # u /= np.linalg.norm(u)
        proj_source = (src.dot(u).astype('long'))
        proj_target = (target.dot(u).astype('long'))

        proj_source_sorted = np.argsort(proj_source)
        proj_target_sorted = np.argsort(proj_target)

        sliced_values[proj_source_sorted] += target[proj_target_sorted]

    sliced_values /= 1

    img = sliced_values[src_label].reshape(origin.shape).astype('uint8')
    return img
def transform_DSW(src,target,src_label,origin,n):
    np.random.seed(1)
    torch.manual_seed(1)
    source_values_cuda = torch.from_numpy(src).float().cuda()
    target_values_cuda = torch.from_numpy(target).float().cuda()
    transform_net = TransformNet(3).cuda()
    op_trannet = optim.Adam(transform_net.parameters(), lr=0.005, betas=(0.5, 0.999))
    for _ in range(10):
        pro = rand_projections(3, n).cuda()
        projections = transform_net(pro)
        cos = cosine_distance_torch(projections, projections)
        reg = 10 * cos
        encoded_projections = source_values_cuda.matmul(projections.transpose(0, 1))
        distribution_projections = (target_values_cuda.matmul(projections.transpose(0, 1)))
        wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                                          torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
        wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, 2), dim=1), 1. / 2)
        wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, 2).mean(), 1. / 2)
        loss = reg - wasserstein_distance
        op_trannet.zero_grad()
        loss.backward()
        op_trannet.step()
    pro = rand_projections(3, n).cuda()
    projections = transform_net(pro).detach().cpu().numpy()
    sliced_values = np.zeros_like(src)
    num_projs = n
    for k in range(num_projs):
        u = projections[k]
        # print(u.shape)
        # u /= np.linalg.norm(u)
        proj_source = (src.dot(u).astype('long'))
        proj_target = (target.dot(u).astype('long'))

        proj_source_sorted = np.argsort(proj_source)
        proj_target_sorted = np.argsort(proj_target)

        sliced_values[proj_source_sorted] += target[proj_target_sorted]

    sliced_values /= num_projs

    img = sliced_values[src_label].reshape(origin.shape).astype('uint8')
    return img
    
def transform_ASW(src,target,src_label,origin,n,p=2,lam=0.01):
    #np.random.seed(1)
    #torch.manual_seed(1)
    source_values_cuda = torch.from_numpy(src).float().cuda()
    target_values_cuda = torch.from_numpy(target).float().cuda()
    phi = Mapping(3).cuda()
    phi_op = optim.Adam(phi.parameters(), lr=0.0001, betas=(0.5, 0.999))
    for _ in range(100):
        first_samples_transform = phi(source_values_cuda)
        second_samples_transform = phi(target_values_cuda)
        
#         reg = lam * (torch.norm(first_samples_transform, p=p, dim=1) + torch.norm(second_samples_transform, p=p,
#                                                                                   dim=1)).mean()
        reg = lam * ((torch.norm(first_samples_transform, p=2, dim=1)**2).mean()**0.5 + (torch.norm(second_samples_transform, p=2,
                                                                                  dim=1)**2).mean()**0.5)
        projections = rand_projections(first_samples_transform.shape[-1], n).to('cuda')
        encoded_projections = first_samples_transform.matmul(projections.transpose(0, 1))
        distribution_projections = second_samples_transform.matmul(projections.transpose(0, 1))
        wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                                          torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
        wasserstein_distance = torch.sum(torch.pow(wasserstein_distance, p), dim=1)
        wasserstein_distance = torch.pow(wasserstein_distance.mean(), 1. / p)
        print('Distance:',wasserstein_distance.item())
        loss = reg - wasserstein_distance
        phi_op.zero_grad()
        loss.backward()
        phi_op.step()
        #temp=torch.ones(source_values_cuda.shape).to('cuda')
        #print(phi(temp))
    first_samples_transform = phi(source_values_cuda).detach().cpu().numpy()
    second_samples_transform = phi(target_values_cuda).detach().cpu().numpy()
    
    pro = rand_projections(6, n).cuda()
    projections = pro.detach().cpu().numpy()
    sliced_values = np.zeros_like(src)
    num_projs = n
    for k in range(num_projs):
        u = projections[k]
        # print(u.shape)
        # u /= np.linalg.norm(u)
        proj_source = (first_samples_transform.dot(u).astype('long'))
        proj_target = (second_samples_transform.dot(u).astype('long'))

        proj_source_sorted = np.argsort(proj_source)
        proj_target_sorted = np.argsort(proj_target)

        sliced_values[proj_source_sorted] += target[proj_target_sorted]

    sliced_values /= num_projs

    img = sliced_values[src_label].reshape(origin.shape).astype('uint8')
    return img


def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections
def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))
def saveDir(x_ori, y_ori, ws=None, wt=None):
    if ws is None:
        ws = np.repeat(1, x_ori.shape[0])
    if wt is None:
        wt = np.repeat(1, y_ori.shape[0])
    pp = x_ori.shape[1]
    data_bind = np.concatenate((x_ori, y_ori))
    weight_bind = np.concatenate((ws, wt))
    data_cov = fastCov(data_bind, weight_bind)
    covinv = np.linalg.inv(data_cov)
    signrt = sqrtm(covinv)

    data_weight = data_bind * weight_bind.reshape(-1, 1)
    cm = data_weight.mean(axis=0)
    # cm = data_bind.mean(axis = 0)
    v1 = fastCov((x_ori - cm) @ signrt, ws)
    v2 = fastCov((y_ori - cm) @ signrt, wt)

    diag = np.diag(np.repeat(1, pp))
    savemat = ((v1 - diag) @ (v1 - diag) + (v2 - diag) @ (v2 - diag)) / 2
    eigenValues, eigenVectors = np.linalg.eig(savemat)
    idx = eigenValues.argsort()[::-1]
    vector = eigenVectors[:, idx[0]]
    dir_temp = signrt @ vector
    return dir_temp / np.sqrt(dir_temp @ dir_temp)
def drDir(x_ori, y_ori, ws=None, wt=None):
    if ws is None:
        ws = np.repeat(1, x_ori.shape[0])
    if wt is None:
        wt = np.repeat(1, y_ori.shape[0])
    pp = x_ori.shape[1]
    data_bind = np.concatenate((x_ori, y_ori))
    weight_bind = np.concatenate((ws, wt))
    data_cov = fastCov(data_bind, weight_bind)
    covinv = np.linalg.inv(data_cov)
    signrt = sqrtm(covinv)

    data_weight = data_bind * weight_bind.reshape(-1, 1)
    cm = data_weight.mean(axis=0)
    # cm = data_bind.mean(axis = 0)
    s1 = (x_ori - cm) @ signrt * ws.reshape(-1, 1)
    s2 = (y_ori - cm) @ signrt * wt.reshape(-1, 1)
    e1 = s1.mean(axis=0)
    e2 = s2.mean(axis=0)
    v1 = fastCov((x_ori - cm) @ signrt, ws)
    v2 = fastCov((y_ori - cm) @ signrt, wt)

    mat1 = ((v1 + np.outer(e1, e1)) @ (v1 + np.outer(e1, e1))
            + (v2 + np.outer(e2, e2)) @ (v2 + np.outer(e2, e2))) / 2
    mat2 = (np.outer(e1, e1) + np.outer(e2, e2)) / 2

    diag = np.diag(np.repeat(1, pp))
    drmat = 2 * mat1 + 2 * mat2 @ mat2 + 2 * sum(np.diag(mat2)) * mat2 - 2 * diag
    eigenValues, eigenVectors = np.linalg.eig(drmat)
    idx = eigenValues.argsort()[::-1]
    vector = eigenVectors[:, idx[0]]
    dir_temp = signrt @ vector
    # dir_temp = signrt@np.linalg.eig(drmat)[1][:,0]
    return dir_temp / np.sqrt(dir_temp @ dir_temp)
def fastCov(data, weight):
    data_weight = data * weight.reshape(-1, 1)
    data_mean = np.mean(data_weight, axis = 0)
    sdata = (data - data_mean)*np.sqrt(weight).reshape(-1, 1)
    data_cov = sdata.T.dot(sdata)/(data.shape[0]-1)
    return data_cov
def transform_drSW(src,target,src_label,origin):
    np.random.seed(1)
    torch.manual_seed(1)
    sliced_values = np.zeros_like(src)


    u = drDir(src,target)
    proj_source = (src.dot(u).astype('long'))
    proj_target = (target.dot(u).astype('long'))

    proj_source_sorted = np.argsort(proj_source)
    proj_target_sorted = np.argsort(proj_target)

    sliced_values[proj_source_sorted] += target[proj_target_sorted]


    img = sliced_values[src_label].reshape(origin.shape).astype('uint8')
    return img
def transform_saveSW(src,target,src_label,origin):
    np.random.seed(1)
    torch.manual_seed(1)
    sliced_values = np.zeros_like(src)


    u = saveDir(src,target)

    proj_source = (src.dot(u).astype('long'))
    proj_target = (target.dot(u).astype('long'))

    proj_source_sorted = np.argsort(proj_source)
    proj_target_sorted = np.argsort(proj_target)

    sliced_values[proj_source_sorted] += target[proj_target_sorted]



    img = sliced_values[src_label].reshape(origin.shape).astype('uint8')
    return img
