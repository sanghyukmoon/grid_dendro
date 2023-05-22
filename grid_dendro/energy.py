import numpy as np
from grid_dendro import dendrogram

def calculate_cumulative_energies(prims, dvol, nodes, node):
    """Calculate cumulative energies as a function of radius in a given node

    Parameters
    ----------
        prims : dict
            dictionary containing primitive variables.
        dvol : float
            representing volume element.
        nodes : dict
            grid_dendro nodes dictionary.
        node : int
            representing the selected node.

    Returns
    -------
        reff : float
            effective radii.
        energies : dict
            thermal, kinetic, gravitational, and total energies
            integrated up to each radius.
    """

    # Create 1-D flattened primitive variables of this node
    req_var = {'rho', 'vel1', 'vel2', 'vel3', 'prs', 'phi'}
    if not req_var.issubset(prims):
        raise ValueError("prims must contain the following variables: {}".format(req_var))

    # Flatten variables
    prims = {k: dendrogram.filter_by_node(v, nodes, node, drop=True)
             for k, v in prims.items()}

     # Sort variables in ascending order of potential
    cells_ordered = prims['phi'].argsort()
    prims = {k: v[cells_ordered] for k, v in prims.items()}


    # Gravitational potential at the HBP boundary
    phimax = prims['phi'][-1]

    # Calculate the center of momentum frame
    rho_cumsum = prims['rho'].cumsum()
    vel1_com = (prims['rho']*prims['vel1']).cumsum() / rho_cumsum
    vel2_com = (prims['rho']*prims['vel2']).cumsum() / rho_cumsum
    vel3_com = (prims['rho']*prims['vel3']).cumsum() / rho_cumsum

    # Kinetic energy
    # \int 0.5 \rho | v - v_com |^2 dV
    # = \int 0.5 \rho |v|^2 dV - (\int 0.5\rho dV) |v_com|^2
    # Note that v_com depends on the limit of the volume integral.
    ekin = (0.5*prims['rho']*(prims['vel1']**2
                              + prims['vel2']**2
                              + prims['vel3']**2)*dvol).cumsum()
    ekin -= (0.5*prims['rho']*dvol).cumsum()*(vel1_com**2 + vel2_com**2 + vel3_com**2)

    # Thermal energy
    gm1 = 5./3. - 1
    ethm = (prims['prs']/gm1*dvol).cumsum()

    # Gravitational energy
    egrv = (prims['rho']*(prims['phi'] - phimax)*dvol).cumsum()

    # Total energy
    etot = ekin + ethm + egrv

    # Effective radius
    Ncells = len(prims['rho'])
    vol = np.full(Ncells, dvol).cumsum()
    reff = (vol / (4.*np.pi/3.))**(1./3.)

    energies = dict(ekin=ekin, ethm=ethm, egrv=egrv, etot=etot)
    return reff, energies


def calculate_cum_energies_old(ds, nodes, node, mode='HBR',
                           boundary_flag='periodic'):
    """WARNING: Deprecated function. Do not use this.

    Do not remove this function for future extension of
    calculate_cumulative_energies.
    """
    # Create 1-D flattened primitive variables of this node
    cells = np.array(nodes[node])
    ds = ds.transpose('z', 'y', 'x')
    dat1d = dict(cells=cells,
                 x=np.broadcast_to(ds.x.data, ds.dens.shape
                                   ).flatten()[cells],
                 y=np.broadcast_to(ds.y.data, ds.dens.shape
                                   ).transpose(1, 2, 0).flatten()[cells],
                 z=np.broadcast_to(ds.z.data, ds.dens.shape
                                   ).transpose(2, 0, 1).flatten()[cells],
                 rho=ds.dens.data.flatten()[cells],
                 vx=ds.vel1.data.flatten()[cells],
                 vy=ds.vel2.data.flatten()[cells],
                 vz=ds.vel3.data.flatten()[cells],
                 prs=ds.prs.data.flatten()[cells],
                 phi=ds.phi.data.flatten()[cells])
    # Assume uniform grid
    dV = ds.dx1*ds.dx2*ds.dx3
    gm1 = (5./3. - 1)
    Ncells = len(cells)

    # Sort variables in ascending order of potential
    cells_sorted = dat1d['phi'].argsort()
    dat1d = {key: value[cells_sorted] for key, value in dat1d.items()}
    cells = dat1d['cells']

    # Gravitational potential at the HBP boundary
    phi0 = dat1d['phi'][-1]

    # Calculate the center of momentum frame
    # note: dV factor is omitted
    M = (dat1d['rho']).cumsum()
    vx0 = (dat1d['rho']*dat1d['vx']).cumsum() / M
    vy0 = (dat1d['rho']*dat1d['vy']).cumsum() / M
    vz0 = (dat1d['rho']*dat1d['vz']).cumsum() / M
    # Potential minimum
    phi_hpr = dendrogram.filter_by_node(ds.phi, cells=cells)
    x0, y0, z0 = get_coords_minimum(phi_hpr)

    # Kinetic energy
    # \int 0.5 \rho | v - v_com |^2 dV
    # = \int 0.5 \rho |v|^2 dV - (\int 0.5\rho dV) |v_com|^2
    # Note that v_com depends on the limit of the volume integral.
    Ekin = (0.5*dat1d['rho']*(dat1d['vx']**2
                              + dat1d['vy']**2
                              + dat1d['vz']**2)*dV).cumsum()
    Ekin -= (0.5*dat1d['rho']*dV).cumsum()*(vx0**2 + vy0**2 + vz0**2)

    # Thermal energy
    Eth = (dat1d['prs']/gm1*dV).cumsum()

    # Gravitational energy
    if mode == 'HBR' or mode == 'HBR+1' or mode == 'HBR-1':
        Egrav = (dat1d['rho']*(dat1d['phi'] - phi0)*dV).cumsum()
    elif mode == 'virial':
        dat1d['gx'] = -ds.phi.differentiate('x').data.flatten()[cells]
        dat1d['gy'] = -ds.phi.differentiate('y').data.flatten()[cells]
        dat1d['gz'] = -ds.phi.differentiate('z').data.flatten()[cells]
        Egrav = (dat1d['rho']*((dat1d['x'] - x0)*dat1d['gx']
                               + (dat1d['y'] - y0)*dat1d['gy']
                               + (dat1d['z'] - z0)*dat1d['gz'])*dV).cumsum()
    else:
        raise ValueError("Unknown mode; select (HBR | HBR+1 | HBR-1 | virial)")

    # Surface terms
    if mode == 'HBR':
        Ekin0 = Eth0 = np.zeros(Ncells)
    elif mode == 'HBR+1' or mode == 'HBR-1':
        pcn = boundary.precompute_neighbor(ds.phi.shape, boundary_flag)
        edge = boundary.get_edge_cells(cells, pcn)
        edg1d = dict(rho=ds.dens.data.flatten()[edge],
                     vx=ds.vel1.data.flatten()[edge],
                     vy=ds.vel2.data.flatten()[edge],
                     vz=ds.vel3.data.flatten()[edge],
                     prs=ds.prs.data.flatten()[edge])
        # COM velocity of edge cells
        M = (edg1d['rho']).sum()
        vx0 = (edg1d['rho']*edg1d['vx']).sum() / M
        vy0 = (edg1d['rho']*edg1d['vy']).sum() / M
        vz0 = (edg1d['rho']*edg1d['vz']).sum() / M
        # Mean surface energies
        Ekin0 = (0.5*edg1d['rho']*((edg1d['vx'] - vx0)**2
                                   + (edg1d['vy'] - vy0)**2
                                   + (edg1d['vz'] - vz0)**2)).mean()
        Eth0 = (edg1d['prs']/gm1).mean()
        # Integrated surface energy to compare with volume energies.
        # Note that the excess energy is given by \int (E - E_0) dV
        Ekin0 = (Ekin0*np.ones(Ncells)*dV).cumsum()
        Eth0 = (Eth0*np.ones(Ncells)*dV).cumsum()
    elif mode == 'virial':
        divPx = ((ds.prs*(ds.x - x0)).differentiate('x')
                 + (ds.prs*(ds.y - y0)).differentiate('y')
                 + (ds.prs*(ds.z - z0)).differentiate('z')
                 ).data.flatten()[cells]
        Eth0 = ((1./3.)*divPx/gm1*dV).cumsum()
        # Kinetic energy surface term
        v0 = np.array([vx0, vy0, vz0])
        # A1
        rho_rdotv = ds.dens*((ds.x - x0)*ds.vel1
                             + (ds.y - y0)*ds.vel2
                             + (ds.z - z0)*ds.vel3)
        A1 = ((rho_rdotv*ds.vel1).differentiate('x')
              + (rho_rdotv*ds.vel2).differentiate('y')
              + (rho_rdotv*ds.vel3).differentiate('z'))
        A1 = (A1.data.flatten()[cells]*dV).cumsum()
        # A2
        grad_rho_r = np.empty((3, 3), dtype=xr.DataArray)
        for i, crd_i in enumerate(['x', 'y', 'z']):
            for j, (crd_j, pos0_j) in enumerate(zip(['x', 'y', 'z'], [x0, y0, z0])):
                grad_rho_r[i, j] = (ds.dens*(ds[crd_j] - pos0_j)
                                    ).differentiate(crd_i)
        A2 = np.empty((3, 3, Ncells))
        for i, crd_i in enumerate(['x', 'y', 'z']):
            for j, (crd_j, pos0_j) in enumerate(zip(['x', 'y', 'z'], [x0, y0, z0])):
                A2[i, j, :] = (grad_rho_r[i, j].data.flatten()[cells]*dV).cumsum()
        A2 = np.einsum('i..., ij..., j...', v0, A2, v0)
        # A3
        grad_rho_rdotv = np.empty(3, dtype=xr.DataArray)
        for i, crd_i in enumerate(['x', 'y', 'z']):
            grad_rho_rdotv[i] = rho_rdotv.differentiate(crd_i)
        A3 = np.empty((3, Ncells))
        for i, crd_i in enumerate(['x', 'y', 'z']):
            A3[i, :] = (grad_rho_rdotv[i].data.flatten()[cells]*dV).cumsum()
        A3 = np.einsum('i...,i...', v0, A3)
        # A4
        div_rhorv = np.empty(3, dtype=xr.DataArray)
        for i, (crd_i, pos0_i) in enumerate(zip(['x', 'y', 'z'],
                                                [x0, y0, z0])):
            div_rhorv[i] = ((ds.dens*(ds[crd_i] - pos0_i)*ds.vel1).differentiate('x')
                            + (ds.dens*(ds[crd_i] - pos0_i)*ds.vel2).differentiate('y')
                            + (ds.dens*(ds[crd_i] - pos0_i)*ds.vel3).differentiate('z'))
        A4 = np.empty((3, Ncells))
        for i, crd_i in enumerate(['x', 'y', 'z']):
            A4[i, :] = (div_rhorv[i].data.flatten()[cells]*dV).cumsum()
        A4 = np.einsum('i...,i...', v0, A4)
        Ekin0 = 0.5*(A1 + A2 - A3 - A4)

    Reff = ((np.ones(Ncells)*dV).cumsum() / (4.*np.pi/3.))**(1./3.)
    if mode == 'HBR':
        Etot = Ekin + Eth + Egrav
    elif mode == 'HBR+1':
        Etot = (Ekin - Ekin0) + (Eth - Eth0) + Egrav
    elif mode == 'HBR-1':
        Etot = (Ekin + Ekin0) + (Eth + Eth0) + Egrav
    elif mode == 'virial':
        Etot = 2*(Ekin - Ekin0) + 3*gm1*(Eth - Eth0) + Egrav

    energies = dict(Reff=Reff, Ekin=Ekin, Eth=Eth, Ekin0=Ekin0, Eth0=Eth0,
                    Egrav=Egrav, Etot=Etot)
    return energies
