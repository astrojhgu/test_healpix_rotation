import numpy as np

from astropy.coordinates import SkyCoord
from astropy.utils.decorators import deprecated_renamed_argument
from healpy import pixelfunc
from healpy import sphtfunc



class Rotator(object):

    def __init__(self, rot=None, coord=None, inv=None, deg=True, eulertype="ZYX"):
        rot_is_seq = hasattr(rot, "__len__") and hasattr(rot[0], "__len__")
        coord_is_seq = (
            hasattr(coord, "__len__")
            and hasattr(coord[0], "__len__")
            and type(coord[0]) is not str
        )
        if rot_is_seq and coord_is_seq:
            if len(rot) != len(coord):
                raise ValueError(Rotator.ErrMessWrongPar)
            else:
                rots = rot
                coords = coord
        elif (rot_is_seq or coord_is_seq) and (rot is not None and coord is not None):
            raise ValueError(Rotator.ErrMessWrongPar)
        else:
            rots = [rot]
            coords = [coord]
        inv_is_seq = hasattr(inv, "__len__")
        if inv_is_seq:
            if len(inv) != len(rots):
                raise ValueError("inv must have same length as rot and/or coord")
            invs = inv
        else:
            invs = [inv] * len(rots)
        # check the argument and normalize them
        if eulertype in ["ZYX", "X", "Y"]:
            self._eultype = eulertype
        else:
            self._eultype = "ZYX"
        self._rots = []
        self._coords = []
        self._invs = []
        for r, c, i in zip(rots, coords, invs):
            rn = normalise_rot(r, deg=deg)
            #            if self._eultype in ['X','Y']:
            #                rn[1] = -rn[1]
            cn = normalise_coord(c)
            self._rots.append(rn)  # append(rn) or insert(0, rn) ?
            self._coords.append(cn)  # append(cn) or insert(0, cn) ?
            self._invs.append(bool(i))
        self._update_matrix()

    def _update_matrix(self):
        self._matrix = np.identity(3)
        self._do_rotation = False
        for r, c, i in zip(self._rots, self._coords, self._invs):
            rotmat, do_rot, rotnorm = get_rotation_matrix(r, eulertype=self._eultype)
            convmat, do_conv, coordnorm = get_coordconv_matrix(c)
            r = np.dot(rotmat, convmat)
            if i:
                r = r.T
            self._matrix = np.dot(self._matrix, r)
            self._do_rotation = self._do_rotation or (do_rot or do_conv)

    def __eq__(self, a):
        if type(a) is not type(self):
            return False
        # compare the _rots
        v = [np.allclose(x, y, rtol=0, atol=1e-15) for x, y in zip(self._rots, a._rots)]
        return (
            np.array(v).all()
            and (self._coords == a._coords)
            and (self._invs == a._invs)
        )

    def __call__(self, *args, **kwds):
        if kwds.pop("inv", False):
            print("a")
            m = self._matrix.T
        else:
            print("b")
            m = self._matrix
        lonlat = kwds.pop("lonlat", False)
        if len(args) == 1:
            arg = args[0]
            if not hasattr(arg, "__len__") or len(arg) < 2 or len(arg) > 3:
                raise TypeError("Argument must be a sequence of 2 or 3 " "elements")
            if len(arg) == 2:
                return rotateDirection(
                    m, arg[0], arg[1], self._do_rotation, lonlat=lonlat
                )
            else:
                return rotateVector(m, arg[0], arg[1], arg[2], self._do_rotation)
        elif len(args) == 2:
            return rotateDirection(
                m, args[0], args[1], self._do_rotation, lonlat=lonlat
            )
        elif len(args) == 3:
            return rotateVector(m, args[0], args[1], args[2], self._do_rotation)
        else:
            raise TypeError("Either 1, 2 or 3 arguments accepted")

    def __mul__(self, a):
        """Composition of rotation."""
        if not isinstance(a, Rotator):
            raise TypeError(
                "A Rotator can only multiply another Rotator "
                "(composition of rotations)"
            )
        rots = self._rots + a._rots
        coords = self._coords + a._coords
        invs = self._invs + a._invs
        return Rotator(rot=rots, coord=coords, inv=invs, deg=False)

    def __rmul__(self, b):
        if not isinstance(b, Rotator):
            raise TypeError(
                "A Rotator can only be multiplied by another Rotator "
                "(composition of rotations)"
            )
        rots = b._rots + self._rots
        coords = b._coords + self._coords
        invs = self._invs + a._invs
        return Rotator(rot=rots, coord=coords, inv=invs, deg=False)

    def __nonzero__(self):
        return self._do_rotation

    def I(self, *args, **kwds):
        """Rotate the given vector or direction using the inverse matrix.
        rot.I(vec) <==> rot(vec,inv=True)
        """
        kwds["inv"] = True
        return self.__call__(*args, **kwds)

    def angle_ref(self, *args, **kwds):
        """Compute the angle between transverse reference direction of initial and final frames
        For example, if angle of polarisation is psi in initial frame, it will be psi+angle_ref in final
        frame.
        Parameters
        ----------
        dir_or_vec : array
          Direction or vector (see Rotator.__call__)
        lonlat: bool, optional
          If True, assume input is longitude,latitude in degrees. Otherwise,
          theta,phi in radian. Default: False
        inv : bool, optional
          If True, use the inverse transforms. Default: False
        Returns
        -------
        angle : float, scalar or array
          Angle in radian (a scalar or an array if input is a sequence of direction/vector)
        """
        R = self
        lonlat = kwds.get("lonlat", False)
        inv = kwds.get("inv", False)
        if len(args) == 1:
            arg = args[0]
            if not hasattr(arg, "__len__") or len(arg) < 2 or len(arg) > 3:
                raise TypeError("Argument must be a sequence of 2 or 3 " "elements")
            if len(arg) == 2:
                v = dir2vec(arg[0], arg[1], lonlat=lonlat)
            else:
                v = arg
        elif len(args) == 2:
            v = dir2vec(args[0], args[1], lonlat=lonlat)
        elif len(args) == 3:
            v = args
        else:
            raise TypeError("Either 1, 2 or 3 arguments accepted")
        vp = R(v, inv=inv)
        north_pole = R([0.0, 0.0, 1.0], inv=inv)
        sinalpha = north_pole[0] * vp[1] - north_pole[1] * vp[0]
        cosalpha = north_pole[2] - vp[2] * np.dot(north_pole, vp)
        return np.arctan2(sinalpha, cosalpha)

    
    
    def rotate_map_pixel(self, m):
        if pixelfunc.maptype(m) == 0:  # a single map is converted to a list
            m = [m]
        npix = len(m[0])
        nside = pixelfunc.npix2nside(npix)
        x_pix_center, y_pix_center, z_pix_center=pixelfunc.pix2vec(nside=nside, ipix=np.arange(npix))

        print(x_pix_center[1000], y_pix_center[1000], z_pix_center[1000])

        x_pix_center_rot, y_pix_center_rot, z_pix_center_rot=self.I(x_pix_center, y_pix_center, z_pix_center)
        print(x_pix_center_rot[1000], y_pix_center_rot[1000], z_pix_center_rot[1000])
        theta_pix_center_rot, phi_pix_center_rot=vec2dir(x_pix_center_rot, y_pix_center_rot, z_pix_center_rot)

        #theta_pix_center_rot, phi_pix_center_rot=vec2dir(self.I(x_pix_center, y_pix_center, z_pix_center))
        # Interpolate the original map to the pixels centers in the new ref frame
        m_rotated = [
            pixelfunc.get_interp_val(each, theta_pix_center_rot, phi_pix_center_rot)
            for each in m
        ]

        # Rotate polarization
        if len(m_rotated) > 1:
            # Create a complex map from QU  and apply the rotation in psi due to the rotation
            # Slice from the end of the array so that it works both for QU and IQU
            L_map = (m_rotated[-2] + m_rotated[-1] * 1j) * np.exp(
                1j * 2 * self.angle_ref(theta_pix_center_rot, phi_pix_center_rot)
            )

            # Overwrite the Q and U maps with the correct values
            m_rotated[-2] = np.real(L_map)
            m_rotated[-1] = np.imag(L_map)
        else:
            m_rotated = m_rotated[0]

        return m_rotated

    def __repr__(self):
        return (
            "[ "
            + ", ".join([str(self._coords), str(self._rots), str(self._invs)])
            + " ]"
        )

    __str__ = __repr__




def rotateVector(rotmat, vec, vy=None, vz=None, do_rot=True):
    """Rotate a vector (or a list of vectors) using the rotation matrix
    given as first argument.
    Parameters
    ----------
    rotmat : float, array-like shape (3,3)
      The rotation matrix
    vec : float, scalar or array-like
      The vector to transform (shape (3,) or (3,N)),
      or x component (scalar or shape (N,)) if vy and vz are given
    vy : float, scalar or array-like, optional
      The y component of the vector (scalar or shape (N,))
    vz : float, scalar or array-like, optional
      The z component of the vector (scalar or shape (N,))
    do_rot : bool, optional
      if True, really perform the operation, if False do nothing.
    Returns
    -------
    vec : float, array
      The component of the rotated vector(s).
    See Also
    --------
    Rotator
    """
    if vy is None and vz is None:
        if do_rot:
            return np.tensordot(rotmat, vec, axes=(1, 0))
        else:
            return vec
    elif vy is not None and vz is not None:
        if do_rot:
            return np.tensordot(rotmat, np.array([vec, vy, vz]), axes=(1, 0))
        else:
            return vec, vy, vz
    else:
        raise TypeError("You must give either vec only or vec, vy " "and vz parameters")


def rotateDirection(rotmat, theta, phi=None, do_rot=True, lonlat=False):
    """Rotate the vector described by angles theta,phi using the rotation matrix
    given as first argument.
    Parameters
    ----------
    rotmat : float, array-like shape (3,3)
      The rotation matrix
    theta : float, scalar or array-like
      The angle theta (scalar or shape (N,))
      or both angles (scalar or shape (2, N)) if phi is not given.
    phi : float, scalar or array-like, optionnal
      The angle phi (scalar or shape (N,)).
    do_rot : bool, optional
      if True, really perform the operation, if False do nothing.
    lonlat : bool
      If True, input angles are assumed to be longitude and latitude in degree,
      otherwise, they are co-latitude and longitude in radians.
    Returns
    -------
    angles : float, array
      The angles of describing the rotated vector(s).
    See Also
    --------
    Rotator
    """
    vx, vy, vz = rotateVector(rotmat, dir2vec(theta, phi, lonlat=lonlat), do_rot=do_rot)
    return vec2dir(vx, vy, vz, lonlat=lonlat)


def vec2dir(vec, vy=None, vz=None, lonlat=False):
    """Transform a vector to angle given by theta,phi.
    Parameters
    ----------
    vec : float, scalar or array-like
      The vector to transform (shape (3,) or (3,N)),
      or x component (scalar or shape (N,)) if vy and vz are given
    vy : float, scalar or array-like, optional
      The y component of the vector (scalar or shape (N,))
    vz : float, scalar or array-like, optional
      The z component of the vector (scalar or shape (N,))
    lonlat : bool, optional
      If True, return angles will be longitude and latitude in degree,
      otherwise, angles will be longitude and co-latitude in radians (default)
    Returns
    -------
    angles : float, array
      The angles (unit depending on *lonlat*) in an array of
      shape (2,) (if scalar input) or (2, N)
    See Also
    --------
    :func:`dir2vec`, :func:`pixelfunc.ang2vec`, :func:`pixelfunc.vec2ang`
    """
    if np.any(np.isnan(vec)):
        return np.nan, np.nan
    if vy is None and vz is None:
        vx, vy, vz = vec
    elif vy is not None and vz is not None:
        vx = vec
    else:
        raise TypeError("You must either give both vy and vz or none of them")
    r = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    ang = np.empty((2, r.size))
    ang[0, :] = np.arccos(vz / r)
    ang[1, :] = np.arctan2(vy, vx)
    if lonlat:
        ang = np.degrees(ang)
        np.negative(ang[0, :], ang[0, :])
        ang[0, :] += 90.0
        return ang[::-1, :].squeeze()
    else:
        return ang.squeeze()


def dir2vec(theta, phi=None, lonlat=False):
    """Transform a direction theta,phi to a unit vector.
    Parameters
    ----------
    theta : float, scalar or array-like
      The angle theta (scalar or shape (N,))
      or both angles (scalar or shape (2, N)) if phi is not given.
    phi : float, scalar or array-like, optionnal
      The angle phi (scalar or shape (N,)).
    lonlat : bool
      If True, input angles are assumed to be longitude and latitude in degree,
      otherwise, they are co-latitude and longitude in radians.
    Returns
    -------
    vec : array
      The vector(s) corresponding to given angles, shape is (3,) or (3, N).
    See Also
    --------
    :func:`vec2dir`, :func:`pixelfunc.ang2vec`, :func:`pixelfunc.vec2ang`
    """
    if phi is None:
        theta, phi = theta
    if lonlat:
        lon, lat = theta, phi
        theta, phi = np.pi / 2.0 - np.radians(lat), np.radians(lon)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    vec = np.empty((3, ct.size), np.float64)
    vec[0, :] = st * cp
    vec[1, :] = st * sp
    vec[2, :] = ct
    return vec.squeeze()






def check_coord(c):
    """Check if parameter is a valid coord system.
    Raise a TypeError exception if it is not, otherwise returns the normalized
    coordinate system name.
    """
    if c is None:
        return c
    if not isinstance(c, str):
        raise TypeError(
            "Coordinate must be a string (G[alactic],"
            " E[cliptic], C[elestial]"
            " or Equatorial=Celestial)"
        )
    if c[0].upper() == "G":
        x = "G"
    elif c[0].upper() == "E" and c != "Equatorial":
        x = "E"
    elif c[0].upper() == "C" or c == "Equatorial":
        x = "C"
    else:
        raise ValueError(
            "Wrong coordinate (either G[alactic],"
            " E[cliptic], C[elestial]"
            " or Equatorial=Celestial)"
        )
    return x


def normalise_coord(coord):
    """Normalise the coord argument.
    Coord sys are either 'E','G', 'C' or 'X' if undefined.
    Input: either a string or a sequence of string.
    Output: a tuple of two strings, each being one of the norm coord sys name
            above.
    eg, 'E' -> ['E','E'], ['Ecliptic','G'] -> ['E','G']
    None -> ['X','X'] etc.
    """
    coord_norm = []
    if coord is None:
        coord = (None, None)
    coord = tuple(coord)
    if len(coord) > 2:
        raise TypeError(
            "Coordinate must be a string (G[alactic],"
            " E[cliptic] or C[elestial])"
            " or a sequence of 2 strings"
        )
    for x in coord:
        coord_norm.append(check_coord(x))
    if len(coord_norm) < 2:
        coord_norm.append(coord_norm[0])
    return tuple(coord_norm)


def normalise_rot(rot, deg=False):
    """Return rot possibly completed with zeroes to reach size 3.
    If rot is None, return a vector of 0.
    If deg is True, convert from degree to radian, otherwise assume input
    is in radian.
    """
    if deg:
        convert = np.pi / 180.0
    else:
        convert = 1.0
    if rot is None:
        rot = np.zeros(3)
    else:
        rot = np.array(rot, np.float64).flatten() * convert
        rot.resize(3, refcheck=False)
    return rot


def get_rotation_matrix(rot, deg=False, eulertype="ZYX"):
    """Return the rotation matrix corresponding to angles given in rot.
    Usage: matrot,do_rot,normrot = get_rotation_matrix(rot)
    Input:
       - rot: either None, an angle or a tuple of 1,2 or 3 angles
              corresponding to Euler angles.
    Output:
       - matrot: 3x3 rotation matrix
       - do_rot: True if rotation is not identity, False otherwise
       - normrot: the normalized version of the input rot.
    """
    rot = normalise_rot(rot, deg=deg)
    if not np.allclose(rot, np.zeros(3), rtol=0.0, atol=1.0e-15):
        do_rot = True
    else:
        do_rot = False
    if eulertype == "X":
        matrot = euler_matrix_new(rot[0], -rot[1], rot[2], X=True)
    elif eulertype == "Y":
        matrot = euler_matrix_new(rot[0], -rot[1], rot[2], Y=True)
    else:
        matrot = euler_matrix_new(rot[0], -rot[1], rot[2], ZYX=True)

    return matrot, do_rot, rot


def get_coordconv_matrix(coord):
    """Return the rotation matrix corresponding to coord systems given
    in coord.
    Usage: matconv,do_conv,normcoord = get_coordconv_matrix(coord)
    Input:
       - coord: a tuple with initial and final coord systems.
                See normalise_coord.
    Output:
       - matconv: the euler matrix for coord sys conversion
       - do_conv: True if matconv is not identity, False otherwise
       - normcoord: the tuple of initial and final coord sys.
    History: adapted from CGIS IDL library.
    """

    coord_norm = normalise_coord(coord)

    if coord_norm[0] == coord_norm[1]:
        matconv = np.identity(3)
        do_conv = False
    else:
        # eps = 23.452294 - 0.0130125 - 1.63889e-6 + 5.02778e-7
        # eps = eps * np.pi / 180.0

        # ecliptic to galactic
        # e2g = np.array(
        #     [
        #         [-0.054882486, -0.993821033, -0.096476249],
        #         [0.494116468, -0.110993846, 0.862281440],
        #         [-0.867661702, -0.000346354, 0.497154957],
        #     ]
        # )

        # ecliptic to equatorial
        # e2q = np.array(
        #     [
        #         [1.0, 0.0, 0.0],
        #         [0.0, np.cos(eps), -1.0 * np.sin(eps)],
        #         [0.0, np.sin(eps), np.cos(eps)],
        #     ]
        # )

        # galactic to ecliptic
        g2e = np.linalg.inv(e2g)

        # galactic to equatorial
        g2q = np.dot(e2q, g2e)

        # equatorial to ecliptic
        q2e = np.linalg.inv(e2q)

        # equatorial to galactic
        q2g = np.dot(e2g, q2e)

        if coord_norm == ("E", "G"):
            matconv = e2g
        elif coord_norm == ("G", "E"):
            matconv = g2e
        elif coord_norm == ("E", "C"):
            matconv = e2q
        elif coord_norm == ("C", "E"):
            matconv = q2e
        elif coord_norm == ("C", "G"):
            matconv = q2g
        elif coord_norm == ("G", "C"):
            matconv = g2q
        else:
            raise ValueError("Wrong coord transform :", coord_norm)
        do_conv = True

    return matconv, do_conv, coord_norm



def euler_matrix_new(a1, a2, a3, X=True, Y=False, ZYX=False, deg=False):
    
    t_k = 0
    if ZYX:
        t_k = t_k + 1
    # if X:   t_k = t_k + 1
    if Y:
        t_k = t_k + 1
    if t_k > 1:
        raise ValueError("Choose either X, Y or ZYX convention")

    convert = 1.0
    if deg:
        convert = np.pi / 180.0

    c1 = np.cos(a1 * convert)
    s1 = np.sin(a1 * convert)
    c2 = np.cos(a2 * convert)
    s2 = np.sin(a2 * convert)
    c3 = np.cos(a3 * convert)
    s3 = np.sin(a3 * convert)

    if ZYX:
        m1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])  # around   z

        m2 = np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])  # around   y

        m3 = np.array([[1, 0, 0], [0, c3, -s3], [0, s3, c3]])  # around   x

    elif Y:
        m1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])  # around   z

        m2 = np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])  # around   y

        m3 = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])  # around   z

    else:
        m1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])  # around   z

        m2 = np.array([[1, 0, 0], [0, c2, -s2], [0, s2, c2]])  # around   x

        m3 = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])  # around   z

    M = np.dot(m3.T, np.dot(m2.T, m1.T))

    return M


