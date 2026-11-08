# Joint model: equations, coordinates, and evidence limits

This experimental calibration model combines bi-prism refraction with the Scheimpflug sensor-plane mapping of a bi-telecentric lens. Its assumptions and validation scope are specified below.

## Checked sources

1. Xin Li et al., *Micro bi-prism bi-telecentric single lens Stereo-DIC system using one-step calibration method*, Optics & Laser Technology 171 (2024) 110303, DOI 10.1016/j.optlastec.2023.110303. Reference for the bi-prism ray path and projection geometry.
2. Yan Hu et al., *Calibration and rectification of bi-telecentric lenses in Scheimpflug condition*, Optics and Lasers in Engineering 149 (2022) 106793, DOI 10.1016/j.optlaseng.2021.106793. Reference for the equivalent telecentric mapping of a tilted sensor plane.

The coordinate, projection, and solver conventions below define this implementation. Relevant changes of variables from the cited papers are stated where needed.

## Source-paper conventions and unified coordinates

Li denotes the world-to-prism extrinsics by R1,T; this implementation uses a rotation R_i and translation t_i for each frame. Li denotes the prism-to-lens orientation by R2; this implementation uses Rp, which affects both refracted ray directions and coordinate transformation. Board coordinates are known, but frame poses are unknown. The first frame is not fixed to the identity pose.

The prism rear plane is z=0, its maximum thickness is t, and the front plane is tau_s*x+z=t. The right face uses tau_R=tan(wedge_R), and the left face uses tau_L=-tan(wedge_L). The physical R face lies at x>=0 and the physical L face at x<=0; z>=0 gives nonnegative glass thickness. Lens +z points toward the object. Both 3D frames are right-handed. The configured omega_p_deg values are the components of a rotation vector in degrees, not XYZ Euler angles. Internally, Rotation.from_rotvec uses radians.

The fixed mapping between lens-reference transverse coordinates and pixel axes is the identity. Optical image inversion is represented by the common signs of j and k. In the example with m<0, both axes reverse; taking the absolute value of magnification would conceal that inversion. Full-sensor pixel coordinates start at the center of the top-left pixel (0,0), with u rightward and v downward. ROI names describe image positions, while the configuration maps each image view to a physical prism branch. The synthetic example deliberately maps image left to physical R.

Forward chain: P → R_i P+t_i → B1_s → B2 → Rp → select xy → D → K_eff → add c.

## Li ray-path implementation

Backward tracing from the lens starts with d0=Rp.T@[0,0,1]. Refraction at the rear air-to-glass plane gives d2; refraction at the front glass-to-air plane gives d1_s. The Snell implementation conserves the tangential direction component, and the normal must point toward the transmitted medium. It detects total internal reflection, grazing incidence, and parallel intersections. Only radicand roundoff outside the valid range by at most 1e-12 is corrected, with a warning.

The homogeneous matrix for projection along d onto a.T X=h is:

    P(d,a,h)[:3,:3] = I - outer(d,a)/(a@d)
    P(d,a,h)[:3,3]  = h*d/(a@d)
    B1_s = P(d1_s,[tau_s,0,1],t)
    B2   = P(d2,[0,0,1],0)
    G_s  = (block_diag(Rp,1) @ B2 @ B1_s)[:2,:]

The rear-plane projection naturally gives z=0; no artificial rear-plane depth is retained. A line-intersection parameter may be negative during backward tracing. The implementation checks physical validity using half-faces, thickness, face order, and the configured aperture.

The independent_trace unit-test path does not call the production Snell routine or B matrices. It independently compares angle-based refraction and two scalar line intersections. Every change in Rp recomputes d0/d2/d1/B/G, and numerical derivative tests cover this dependency.

## Hu Scheimpflug mapping and the joint assumption

Hu defines the sensor rotation as Rx(alpha)Ry(beta). Intersecting parallel chief rays with the tilted sensor plane gives:

    S = [[sec(beta), 0], [-tan(alpha)*tan(beta), sec(alpha)]]
    K_eff = [[j,0],[l,k]]

For square pixels and a common magnification, j=m/cos(beta), k=m/cos(alpha), and l=-m*tan(alpha)*tan(beta). When a known nonsquare pixel geometry is used, the code left-multiplies S by diag(1,pixel_u/pixel_v); m_px_per_mm is defined relative to the u pixel size. Mechanical angles are not exported if the pixel-size ratio is unknown.

Connecting Hu's sensor mapping after Li's prism ray model is the joint-model assumption. Hu's multipose linear initializer for a bare bi-telecentric lens is not applied directly to raw prism images.

Further assumptions are an ideal bi-telecentric lens; ideal planar prism front and rear surfaces with ridge and side edges parallel; a prism rear-surface origin on the lens axis, without an independent transverse translation; and n specified for the configured wavelength or as an effective refractive index. Dispersion, PSF asymmetry, and manufacturing errors are not modeled.

## Two algebraic and dimensional corrections

The u vector in Li's Eq. (7) does not explicitly include the principal point. If xp,yp are raw pixels, u0,v0 must first be subtracted. This implementation initializes by solving K against p-c and then subtracting b_s; it does not assume centered input pixels.

The printed right-hand side of Li's Eq. (10) applies (H.T H)^-1 to a four-component vector even though H is 4×3, which is dimensionally inconsistent. The least-squares solution is X=H^+b, or equivalently (H.T H)^-1H.T b when H has full column rank. The implementation uses numpy.linalg.lstsq/SVD instead of explicitly inverting normal equations.

Hu's choice c=0 for an undistorted initial solution is a coordinate choice. The joint model fixes c to user-provided full-sensor coordinates by default to reduce coupling with pose and scale. An assumed image center is not reported as a measured one.

## Distortion assumptions

    r2=(qx^2+qy^2)/q_scale_mm^2
    D(q)=q*(1+kappa1*r2+kappa2*r2^2)

q_scale is fixed, and kappa coefficients are dimensionless. D is evaluated in lens-reference millimeter coordinates, then multiplied by K and shifted by c. This differs from Li's five-coefficient radial/tangential model and Hu's three-term radial model; their coefficient values cannot be copied into this model. The implementation does not claim that Scheimpflug shear and general nonlinear tangential distortion are strictly equivalent over the full field.

For reconstruction, the radial inverse checks that the derivative of r_d=r(1+k1 r²/s²+k2 r⁴/s⁴) is positive over the declared field and uses brentq to find the unique radius. Inputs outside the declared invertible domain, nonmonotonic distortion, or rank-deficient reconstruction H are rejected.

## Initialization and optimization

With provisional K and Rp, the left and right p-c values are first mapped to reference coordinates. Within each frame, the stacked design matrix [X*M_s,Y*M_s,M_s] is used to solve linearly for r1,r2,t. For nonplanar points, the matrix is [X*M_s,Y*M_s,Z*M_s,M_s]. Full rank is checked, the rotation is projected onto SO(3) by SVD, and poses are then optimized.

Stage A holds global parameters fixed and optimizes extrinsics. Stage B jointly optimizes K/Rp/extrinsics with zero distortion. Stage C releases kappa according to the configuration. Defaults are trf, soft_l1, and sparse finite differences. Parameters have scaling and bounds; j/k use log magnitudes with fixed signs. The number of observations stays fixed during each solve. An invalid path receives a large fixed-length residual penalty with its cause recorded, rather than deleting a residual. Final reprojection and physical ray-path checks are also run without penalties.

Reliable, known covariance may be supplied in CSV with covariance_status=validated; use_covariance=true enables Cholesky whitening. The detector does not estimate reliable center covariance or report unjustified precision. Euclidean image-point RMS, coordinate-component RMS, whitened squared sum, and robust objective are reported separately.

## Identifiability and angle export

The data Jacobian uses central differences of raw pixel residuals with respect to scaled parameters, without priors or robust weights. A whitened/prior-augmented Jacobian is reported separately. After projecting out the column space of frame extrinsics, the global columns are checked for singular values, rank, weak directions, and column correlation. Numerical full rank is only a local test for the current data, parameter scaling, and threshold; it does not establish measured mechanical accuracy.

The effective mode exports K, Rp, and matrices needed for reconstruction; mechanical angles remain null. The physical_constrained mode is a conservative export path and does not freely fit an additional m/alpha/beta triplet. The first version requires an independently measured and fixed Rp, known pixel-size ratio, the assembly sign of alpha, and passing numerical identifiability and boundary checks. Only then is the physical root of cos²alpha obtained from Hu's Eq. (23), followed by m/beta. Information on the opposite sign branch is retained. Near-zero angles or multiple solutions do not yield a unique mechanical angle. The effective model remains usable without mechanical angles, subject to review of the calibration report.

Training and validation are split by complete frames. For each held-out frame, about half the point IDs fit its pose; the remaining IDs evaluate pixel error and known distances on the board. Free-scale alignment is not allowed. Independent translation validation requires an external CSV; fitted poses are not treated as ground truth.
