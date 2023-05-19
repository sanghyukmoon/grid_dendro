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
