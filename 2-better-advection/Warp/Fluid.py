import warp as wp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim

wp.init()
# wp.config.verify_cuda = True

grid_width = wp.constant(400)
grid_height = wp.constant(400)
density = wp.constant(0.5)
hx = wp.constant(1.0/min(grid_width.val, grid_height.val))


@wp.func
def at(f: wp.array2d(dtype=float),
       x: int,
       y: int):

    x = wp.clamp(x, 0, grid_width-1)
    y = wp.clamp(y, 0, grid_height-1)

    return f[x, y]

@wp.func
def length(x: float, y: float):
    return wp.sqrt(x*x + y*y)

@wp.func
def cubicPulse(x: float):
    x = min(wp.abs(x), 1.0)
    return 1.0 - x*x*(3.0-2.0*x)


@wp.kernel
def addInflow(d_src: wp.array2d(dtype=float),
              u_src: wp.array2d(dtype=float),
              v_src: wp.array2d(dtype=float), d: float, u: float, v: float):

    i, j = wp.tid()

    x0 = 20
    x1 = 25
    y0 = 180
    y1 = 220

    if i < x1 and i > x0 and j < y1 and j > y0:

        # l = length((2.0*(float(i) + 0.5)*hx - (float(x0)*hx + float(x1)*hx))/(float(x1)*hx - float(x0)*hx),
        #             (2.0*(float(j) + 0.5)*hx - (float(y0)*hx + float(y1)*hx))/(float(y1)*hx - float(y0)*hx))

        l = length((2.0*(float(i) + 0.5) - (float(x0) + float(x1)))/(float(x1) - float(x0)),
                    (2.0*(float(j) + 0.5) - (float(y0) + float(y1)))/(float(y1) - float(y0)))

        vi = cubicPulse(l)
 
        if d_src[i, j] < d:
            d_src[i, j]  = vi * d
        if u_src[i, j] < u:
            u_src[i, j]  = vi * u
        if v_src[i, j] < v:
            v_src[i, j]  = vi * v


@wp.func
def cerp(a: float, b: float, c: float, d: float, x: float):
    xsq = x*x
    xcu = xsq*x

    minV = min(a, min(b, min(c, d)))
    maxV = max(a, max(b, max(c, d)))

    t = a*(0.0 - 0.5*x + 1.0*xsq - 0.5*xcu) + \
        b*(1.0 + 0.0*x - 2.5*xsq + 1.5*xcu) + \
        c*(0.0 + 0.5*x + 2.0*xsq - 1.5*xcu) + \
        d*(0.0 + 0.0*x - 0.5*xsq + 0.5*xcu)
    
    return min(max(t, minV), maxV)


@wp.func
def cubic_grid_sample(fluid: wp.array2d(dtype=float), x: float, y: float, ox: float, oy: float):
    x_new = min(max(x - ox, 0.0), float(grid_width) - 1.001)
    y_new = min(max(y - oy, 0.0), float(grid_height) - 1.001)
    ix = int(x_new)
    iy = int(y_new)
    x_new = x_new - float(ix)
    y_new = y_new - float(iy)

    x0 = max(ix - 1, 0)
    x1 = ix
    x2 = ix + 1
    x3 = min(ix + 2, grid_width - 1)

    y0 = max(iy - 1, 0)
    y1 = iy
    y2 = iy + 1
    y3 = min(iy + 2, grid_height - 1)
    
    q0 = cerp(at(fluid, x0, y0), at(fluid, x1, y0), at(fluid, x2, y0), at(fluid, x3, y0), x)
    q1 = cerp(at(fluid, x0, y1), at(fluid, x1, y1), at(fluid, x2, y1), at(fluid, x3, y1), x)
    q2 = cerp(at(fluid, x0, y2), at(fluid, x1, y2), at(fluid, x2, y2), at(fluid, x3, y2), x)
    q3 = cerp(at(fluid, x0, y3), at(fluid, x1, y3), at(fluid, x2, y3), at(fluid, x3, y3), x)
    
    return cerp(q0, q1, q2, q3, y)


@wp.func
def lin_grid_sample(fluid: wp.array2d(dtype=float), x: float, y: float, ox: float, oy: float):
    x_new = min(max(x - ox, 0.0), float(grid_width) - 1.001)
    y_new = min(max(y - oy, 0.0), float(grid_height) - 1.001)
    ix = int(x_new)
    iy = int(y_new)
    x_new = x_new - float(ix)
    y_new = y_new - float(iy)

    x00 = at(fluid, ix + 0, iy + 0)
    x10 = at(fluid, ix + 1, iy + 0)
    x01 = at(fluid, ix + 0, iy + 1)
    x11 = at(fluid, ix + 1, iy + 1)

    return wp.lerp(wp.lerp(x00, x10, x_new), wp.lerp(x01, x11, x_new), y_new)


@wp.func
def rungeKutta3(x: float, y: float, timestep: float, u: wp.array2d(dtype=float), v: wp.array2d(dtype=float)):
    firstU = lin_grid_sample(u, x, y, 0.0, 0.5)/hx
    firstV = lin_grid_sample(v, x, y, 0.5, 0.0)/hx

    midX = x - 0.5*timestep*firstU
    midY = y - 0.5*timestep*firstV

    midU = lin_grid_sample(u, midX, midY, 0.0, 0.5)/hx
    midV = lin_grid_sample(v, midX, midY, 0.5, 0.0)/hx

    lastX = x - 0.75*timestep*midU
    lastY = y - 0.75*timestep*midV

    lastU = lin_grid_sample(u, lastX, lastY, 0.0, 0.5)
    lastV = lin_grid_sample(v, lastX, lastY, 0.5, 0.0)

    x = x - (timestep*((2.0/9.0)*firstU + (3.0/9.0)*midU + (4.0/9.0)*lastU))
    y = y - (timestep*((2.0/9.0)*firstV + (3.0/9.0)*midV + (4.0/9.0)*lastV))

    return wp.vec2(x, y)


@wp.kernel
def advect(d_src: wp.array2d(dtype=float),
           d_dst: wp.array2d(dtype=float),
           u_src: wp.array2d(dtype=float),
           u_dst: wp.array2d(dtype=float),
           v_src: wp.array2d(dtype=float),
           v_dst: wp.array2d(dtype=float),
           dt: float):

    i, j = wp.tid()

    x = float(i)
    y = float(j)

    _ = wp.vec2()

    _ = rungeKutta3(x, y, dt, u_src, v_src)

    x = _[0]
    y = _[1]

    d_dst[i, j] = cubic_grid_sample(d_src, x, y, 0.5, 0.5)
    u_dst[i, j] = cubic_grid_sample(u_src, x, y, 0.0, 0.5)
    v_dst[i, j] = cubic_grid_sample(v_src, x, y, 0.5, 0.0)


@wp.kernel
def buildRhs(r: wp.array2d(dtype=float), u: wp.array2d(dtype=float), v: wp.array2d(dtype=float)):
    scale = 1.0/hx
    i, j = wp.tid()
    r[i, j] = -scale*(at(u, i+1, j) - at(u, i, j) +
                      at(v, i, j+1) - at(v, i, j))


@wp.kernel
def project(r: wp.array2d(dtype=float), p_src: wp.array2d(dtype=float), p_dst: wp.array2d(dtype=float), timestep: float):

    scale = timestep / (density*hx*hx)

    i, j = wp.tid()

    diag = float(0.0)
    offDiag = float(0.0)

    if i > 0:
        diag += scale
        offDiag -= scale * at(p_src, i-1, j)
    if j > 0:
        diag += scale
        offDiag -= scale * at(p_src, i, j-1)
    if i < grid_width - 1:
        diag += scale
        offDiag -= scale * at(p_src, i+1, j)
    if j < grid_height - 1:
        diag += scale
        offDiag -= scale * at(p_src, i, j+1)

    newP = (at(r, i, j) - offDiag)/diag
    p_dst[i, j] = newP


@wp.kernel
def applyPressure(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), p: wp.array2d(dtype=float), timestep: float):

    scale = timestep / (density * hx)
    i, j = wp.tid()

    u[i, j] = u[i, j] - scale*p[i, j]
    u[(i+1), j] = u[(i+1), j] + scale*p[i, j]
    v[i, j] = v[i, j] - scale*p[i, j]
    v[i, (j+1)] = v[i, (j+1)] + scale*p[i, j]

    if i == 0 or i == grid_width:
        u[i, j] = 0.0
    if j == 0 or j == grid_height:
        v[i, j] = 0.0


class FluidSolver:

    def __init__(self):

        self.sim_substeps = 1
        self.sim_dt = 0.01
        self.sim_time = 0.0
        self.iterations = 600

        self.device = wp.get_device()

        shape = (grid_width.val, grid_height.val)

        self.u_src = wp.zeros(shape, dtype=float)
        self.u_dst = wp.zeros(shape, dtype=float)

        self.v_src = wp.zeros(shape, dtype=float)
        self.v_dst = wp.zeros(shape, dtype=float)

        self.d_src = wp.zeros(shape, dtype=float)
        self.d_dst = wp.zeros(shape, dtype=float)

        self.r = wp.zeros(shape, dtype=float)
        self.p_src = wp.zeros(shape, dtype=float)
        self.p_dst = wp.zeros(shape, dtype=float)

    def render(self, img, i):

        for i in range(self.sim_substeps):

            shape = (grid_width.val, grid_height.val)
            dt = self.sim_dt

            # update emitters
            wp.launch(addInflow, dim=shape, inputs=[
                      self.d_src, self.u_src, self.v_src, 1.0, 0.0, 3.0])

            wp.launch(buildRhs, dim=shape, inputs=[
                      self.r, self.u_src, self.v_src])

            for j in range(self.iterations):
                wp.launch(project, dim=shape, inputs=[self.r, self.p_src, self.p_dst, dt])
                (self.p_src, self.p_dst) = (self.p_dst, self.p_src)

            wp.launch(applyPressure, dim=shape, inputs=[
                      self.u_src, self.v_src, self.p_src, dt])

            # advect
            wp.launch(advect, dim=shape, inputs=[
                      self.d_src, self.d_dst, self.u_src, self.u_dst, self.v_src, self.v_dst, dt])

            (self.d_src, self.d_dst) = (self.d_dst, self.d_src)
            (self.u_src, self.u_dst) = (self.u_dst, self.u_src)
            (self.v_src, self.v_dst) = (self.v_dst, self.v_src)

            self.sim_time += dt

        img.set_array(self.d_src.numpy())

        return img,


if __name__ == '__main__':

    chapter2 = FluidSolver()

    fig = plt.figure()

    img = plt.imshow(chapter2.d_src.numpy(), origin="lower", animated=True, interpolation="antialiased")
    img.set_norm(matplotlib.colors.Normalize(0.0, 1.0))
    seq = anim.FuncAnimation(fig, lambda i: chapter2.render(img, i), frames=10000, blit=True, interval=8, repeat=False)
    # seq.save('chapter2.gif') 
    
    plt.show()