"""Extended Kalman filter: 3D drone position, velocity and size from a moving monocular camera.

STATE (13 numbers)
    x = [ p (3),  v (3),  S,  b_t (3),  b_th (3) ]
    p    drone position, world frame [m]
    v    drone velocity, world frame [m/s]
    S    drone's physical width [m]                       (random constant)
    b_t  slowly drifting error of the GPS camera position [m]          (Gauss-Markov)
    b_th slowly drifting error of the IMU camera attitude [rad]         (Gauss-Markov)

WHY S IS A STATE (and not just extra measurement noise)
    The box width only tells us S / Z (size over depth), see the measurement model below.
    We do not know S exactly: the prior is S ~ N(0.35, 0.15^2) m from public specs.
    If we plugged in S = 0.35 and called the error "measurement noise", that noise would be
    the SAME every frame (it is one fixed, unknown drone), but a Kalman filter assumes
    measurement noise is independent between frames. It would then average 100 frames and
    believe the size error shrank by sqrt(100) = 10x, which is false: an overconfident filter.
    Putting S in the state as a "random constant" (no process noise, initial variance 0.15^2)
    models this correctly: the uncertainty about S stays until something else (camera
    parallax) actually tells us the range, and then the filter can learn S too.
    The camera pose errors b_t, b_th are states for exactly the same reason: GPS/INS errors
    drift slowly, so they are nearly the same in consecutive frames (see sim/noise.py).

MOTION MODEL
    p_{k+1} = p_k + dt v_k              nearly constant velocity: v changes only by a
    v_{k+1} = v_k + w_v                 random (white) acceleration with density q
    S_{k+1} = S_k                       random constant
    b_{k+1} = a b_k + w_b               Gauss-Markov drift, a = exp(-dt/tau)
    -> x_{k+1} = F x_k + w,  w ~ N(0, Q)

MEASUREMENT MODEL (one detection; GPS/INS reports the camera pose R_meas, t_meas)
    corrected pose:  t = t_meas - b_t,   R = R_meas @ Rot(-b_th)
    p_c = R^T (p - t) = (X, Y, Z)        drone in camera coordinates
    u   = fx * X / Z + cx                box centre [px]
    v   = fy * Y / Z + cy
    l   = log(w_px) = log(fx) + log(S) - log(Z)      log of the box width
    We use log(width) instead of width because the width noise is multiplicative
    (a percentage); in log space it is additive Gaussian noise, which is what a Kalman
    filter assumes. It carries the same information as the width itself.
    Measurement noise R = detector noise + the small WHITE part of the pose noise.
"""
import numpy as np

from sim.camera import rotation_from_vector, skew
from sim.noise import PoseNoise

# index of each block inside the 13-element state
IP = slice(0, 3)     # position
IV = slice(3, 6)     # velocity
IS = 6               # size
IBT = slice(7, 10)   # camera position drift
IBR = slice(10, 13)  # camera attitude drift
NX = 13


# ------------------------------------------------------------------ motion model
def cv_blocks(dt, q):
    """6x6 F and Q for [p, v] under white acceleration noise of density q [m^2/s^3].
    Per axis, integrating a white acceleration over dt gives
        Var(p) = q dt^3/3,   Cov(p, v) = q dt^2/2,   Var(v) = q dt."""
    I3 = np.eye(3)
    F = np.block([[I3, dt * I3], [np.zeros((3, 3)), I3]])
    Q = q * np.block([[dt**3 / 3 * I3, dt**2 / 2 * I3], [dt**2 / 2 * I3, dt * I3]])
    return F, Q


def gm_coeff(dt, tau):
    """Gauss-Markov decay a = exp(-dt/tau); tau = 0 means white noise (a = 0)."""
    return np.exp(-dt / tau) if tau > 0 else 0.0


# ------------------------------------------------------------------ measurement model
def corrected_pose(x, R_meas, t_meas):
    """Best estimate of the TRUE camera pose: remove the estimated drift."""
    return R_meas @ rotation_from_vector(-x[IBR]), t_meas - x[IBT]


def measurement(x, R_meas, t_meas, K, use_width=True):
    """h(x): predicted [u, v, log w] (or [u, v] for the bearing-only filter)."""
    R, t = corrected_pose(x, R_meas, t_meas)
    X, Y, Z = R.T @ (x[IP] - t)
    u = K.fx * X / Z + K.cx
    v = K.fy * Y / Z + K.cy
    if not use_width:
        return np.array([u, v])
    return np.array([u, v, np.log(K.fx) + np.log(x[IS]) - np.log(Z)])


def d_h_d_pc(p_c, K, use_width=True):
    """Derivative of [u, v, log w] with respect to the camera-frame point (X, Y, Z):
        du/dX = fx/Z,  du/dZ = -fx X/Z^2,  dv/dY = fy/Z,  dv/dZ = -fy Y/Z^2,  dl/dZ = -1/Z."""
    X, Y, Z = p_c
    J = np.array([[K.fx / Z, 0.0, -K.fx * X / Z**2],
                  [0.0, K.fy / Z, -K.fy * Y / Z**2],
                  [0.0, 0.0, -1.0 / Z]])
    return J if use_width else J[:2]


def measurement_jacobian(x, R_meas, t_meas, K, use_width=True):
    """H = dh/dx (analytic). Chain rule through p_c = R^T (p - t_meas + b_t):
        dp_c/dp    = R^T
        dp_c/db_t  = R^T                         (same as p: only p - t matters)
        dp_c/db_th = -skew(p_c)                  (small rotation: Rot(d) p_c ~ p_c + d x p_c)
    v does not appear in h (zero columns). Only log w depends on S: d(log S)/dS = 1/S."""
    R, t = corrected_pose(x, R_meas, t_meas)
    p_c = R.T @ (x[IP] - t)
    J = d_h_d_pc(p_c, K, use_width)
    H = np.zeros((J.shape[0], NX))
    H[:, IP] = J @ R.T
    H[:, IBT] = J @ R.T
    H[:, IBR] = J @ (-skew(p_c))
    if use_width:
        H[2, IS] = 1.0 / x[IS]
    return H


# ------------------------------------------------------------------ the filter
class DroneEKF:
    """Per frame:  ekf.predict(dt);  ekf.update(z, R_det, R_meas, t_meas)."""

    def __init__(self, K, q=0.1, size_prior=(0.35, 0.15), sigma_v0=5.0, pose_noise=None,
                 use_width=True, n_iter=3, min_size=0.05):
        self.K = K
        self.q = q                               # process noise density [m^2/s^3]
        self.S0, self.sigma_S0 = size_prior      # prior on the drone's width [m]
        self.sigma_v0 = sigma_v0                 # initial velocity std per axis [m/s]
        self.pn = pose_noise or PoseNoise()      # what the filter assumes about GPS/INS errors
        self.use_width = use_width               # False = bearing-only filter
        self.n_iter = n_iter                     # 1 = standard EKF, >1 = iterated EKF update
        self.min_size = min_size                 # keep S > 0 (h uses log S)
        self.x = None
        self.P = None

    # --- initialisation from the first detection
    def initialize(self, z, R_det, R_meas, t_meas):
        """Back-project the first detection. Direction from the pixel (u, v); depth from the
        box width and the size PRIOR:  Z0 = fx * S0 / w.
            p0 = t_meas + R_meas [m_x Z0, m_y Z0, Z0],   m_x = (u-cx)/fx,  m_y = (v-cy)/fy
        Covariance: P0 = J Sigma J^T, J = d(state)/d(u, v, log w, S, b_t, b_th), Sigma = their
        variances. The S column couples position and size (a bigger drone is farther away
        along the ray), which gives a long, thin uncertainty ellipsoid ALONG THE LINE OF
        SIGHT: range is uncertain by ~40%, the direction is not. Velocity starts at 0 with
        std sigma_v0 per axis. All variants (also bearing-only) start this way."""
        K, pn = self.K, self.pn
        u, v, w = z
        m = np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0])
        Z0 = K.fx * self.S0 / w
        p0 = t_meas + R_meas @ (m * Z0)

        # J: rows = state (13), columns = inputs [u, v, log w, S, b_t (3), b_th (3)]
        J = np.zeros((NX, 10))
        J[IP, 0] = R_meas @ np.array([Z0 / K.fx, 0.0, 0.0])
        J[IP, 1] = R_meas @ np.array([0.0, Z0 / K.fy, 0.0])
        J[IP, 2] = -R_meas @ (m * Z0)               # Z0 = fx S exp(-log w)
        J[IP, 3] = R_meas @ (m * Z0 / self.S0)      # dZ0/dS = Z0/S
        J[IP, 4:7] = -np.eye(3)                     # true camera = t_meas - b_t
        J[IP, 7:10] = R_meas @ skew(m * Z0)         # true attitude = R_meas Rot(-b_th)
        J[IS, 3] = 1.0
        J[IBT, 4:7] = np.eye(3)
        J[IBR, 7:10] = np.eye(3)
        Sigma = np.diag([R_det[0, 0] + (K.fx * pn.sigma_att_white) ** 2,
                         R_det[1, 1] + (K.fy * pn.sigma_att_white) ** 2,
                         R_det[2, 2], self.sigma_S0**2]
                        + [pn.sigma_pos**2] * 3 + [pn.sigma_att**2] * 3)
        self.x = np.zeros(NX)
        self.x[IP], self.x[IS] = p0, self.S0
        self.P = J @ Sigma @ J.T
        self.P[IV, IV] = self.sigma_v0**2 * np.eye(3)
        self.P[IP, IP] += pn.sigma_pos_white**2 * np.eye(3)

    # --- predict
    def predict(self, dt):
        """x <- F x,   P <- F P F^T + Q"""
        pn = self.pn
        a_t, a_r = gm_coeff(dt, pn.tau_pos), gm_coeff(dt, pn.tau_att)
        F, Q = np.eye(NX), np.zeros((NX, NX))
        F[0:6, 0:6], Q[0:6, 0:6] = cv_blocks(dt, self.q)
        F[IBT, IBT] = a_t * np.eye(3)
        F[IBR, IBR] = a_r * np.eye(3)
        Q[IBT, IBT] = (1 - a_t**2) * pn.sigma_pos**2 * np.eye(3)   # keeps Var(b) = sigma^2
        Q[IBR, IBR] = (1 - a_r**2) * pn.sigma_att**2 * np.eye(3)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # --- update
    def measurement_noise(self, R_det, H):
        """R = detector noise + white pose jitter mapped to pixels through H.
        (The b_t / b_th columns of H are exactly dh/d(position error), dh/d(attitude error).)"""
        pn = self.pn
        m = H.shape[0]
        return (R_det[:m, :m]
                + pn.sigma_pos_white**2 * H[:, IBT] @ H[:, IBT].T
                + pn.sigma_att_white**2 * H[:, IBR] @ H[:, IBR].T)

    def update(self, z, R_det, R_meas, t_meas):
        """EKF update with one detection z = [u, v, w_px] (w_px unused if bearing-only).
        With n_iter > 1 this is the iterated EKF: re-linearise h around the updated estimate
        and redo the update (Gauss-Newton). It helps when the first linearisation point is
        far off, e.g. range wrong by 40% right after initialisation.
        Returns (innovation, innovation covariance, NIS), all at the prior estimate."""
        m = 3 if self.use_width else 2
        z_vec = np.array([z[0], z[1], np.log(z[2])])[:m]
        x_prior, P = self.x, self.P
        x_i = x_prior.copy()
        for it in range(self.n_iter):
            H = measurement_jacobian(x_i, R_meas, t_meas, self.K, self.use_width)
            R = self.measurement_noise(R_det, H)
            # innovation, linearised around x_i (for it = 0, x_i = x_prior: standard EKF)
            y = z_vec - measurement(x_i, R_meas, t_meas, self.K, self.use_width) - H @ (x_prior - x_i)
            S_inn = H @ P @ H.T + R                     # innovation covariance
            Kg = P @ H.T @ np.linalg.inv(S_inn)         # Kalman gain
            if it == 0:
                y0, S0 = y, S_inn
            x_i = x_prior + Kg @ y
            x_i[IS] = max(x_i[IS], self.min_size)       # S must stay > 0 because h uses log S
        self.x = x_i
        # Joseph form: (I-KH) P (I-KH)^T + K R K^T. Same as (I-KH)P in exact arithmetic,
        # but stays symmetric and positive definite despite rounding.
        I_KH = np.eye(NX) - Kg @ H
        self.P = I_KH @ P @ I_KH.T + Kg @ R @ Kg.T
        return y0, S0, float(y0 @ np.linalg.solve(S0, y0))

    # --- outputs
    def range_estimate(self, t_meas):
        """Range r = |p - t| from the corrected camera position t = t_meas - b_t, and its std.
        Linearised: dr/dp = d, dr/db_t = d (d = unit line of sight), so
            sigma_r^2 = [d, d] P_{(p, b_t)} [d, d]^T + white GPS jitter.
        The drift b_t moves our world estimate of the drone and of the camera together, so
        it largely cancels in the range: the strong negative p/b_t covariance handles that."""
        diff = self.x[IP] - (t_meas - self.x[IBT])
        r = float(np.linalg.norm(diff))
        J = np.zeros(NX)
        J[IP] = diff / r
        J[IBT] = diff / r
        var = float(J @ self.P @ J) + self.pn.sigma_pos_white**2
        return r, np.sqrt(var)
