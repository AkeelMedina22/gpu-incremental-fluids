import warp as wp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim

wp.init()

grid_width = wp.constant(400)
grid_height = wp.constant(400)
density = wp.constant(0.01)
hx = wp.constant(1.0/min(grid_width.val, grid_height.val))


@wp.struct
class Fluid:
    w   : int
    h   : int
    ox  : float
    oy  : float
    src : wp.array2d(dtype=float)
    dst : wp.array2d(dtype=float)


@wp.func
def at(f: wp.array2d(dtype=float),
                 x: int,
                 y: int):

    x = wp.clamp(x, 0, grid_width-1)
    y = wp.clamp(y, 0, grid_height-1)

    return f[x,y]
    

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

    x0 = 180
    x1 = 220
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
def lin_grid_sample(f: Fluid, x: float, y: float):

    x = min(max(x - f.ox, 0.0), float(f.w) - 1.001)
    y = min(max(y - f.oy, 0.0), float(f.h) - 1.001)
    ix = int(x)
    iy = int(y)
    x = x - float(ix)
    y = y - float(iy)
    
    x00 = at(f.src, ix + 0, iy + 0)
    x10 = at(f.src, ix + 1, iy + 0)
    x01 = at(f.src, ix + 0, iy + 1)
    x11 = at(f.src, ix + 1, iy + 1)
    
    return wp.lerp(wp.lerp(x00, x10, x), wp.lerp(x01, x11, x), y)


@wp.func
def cubic_grid_sample(f: Fluid, x: float, y: float):
    x = min(max(x - f.ox, 0.0), float(f.w) - 1.001)
    y = min(max(y - f.oy, 0.0), float(f.h) - 1.001)
    ix = int(x)
    iy = int(y)
    x = x - float(ix)
    y = y - float(iy)

    x0 = max(ix - 1, 0)
    x1 = ix
    x2 = ix + 1
    x3 = min(ix + 2, grid_width - 1)

    y0 = max(iy - 1, 0)
    y1 = iy
    y2 = iy + 1
    y3 = min(iy + 2, grid_height - 1)
    
    q0 = cerp(at(f.src, x0, y0), at(f.src, x1, y0), at(f.src, x2, y0), at(f.src, x3, y0), x)
    q1 = cerp(at(f.src, x0, y1), at(f.src, x1, y1), at(f.src, x2, y1), at(f.src, x3, y1), x)
    q2 = cerp(at(f.src, x0, y2), at(f.src, x1, y2), at(f.src, x2, y2), at(f.src, x3, y2), x)
    q3 = cerp(at(f.src, x0, y3), at(f.src, x1, y3), at(f.src, x2, y3), at(f.src, x3, y3), x)
    
    return cerp(q0, q1, q2, q3, y)


@wp.func
def rungeKutta3(u: Fluid, v: Fluid, x: float, y: float, timestep: float):
    firstU = lin_grid_sample(u, x, y)/hx
    firstV = lin_grid_sample(v, x, y)/hx

    midX = x - 0.5*timestep*firstU
    midY = y - 0.5*timestep*firstV

    midU = lin_grid_sample(u, midX, midY)/hx
    midV = lin_grid_sample(v, midX, midY)/hx

    lastX = x - 0.75*timestep*midU
    lastY = y - 0.75*timestep*midV

    lastU = lin_grid_sample(u, lastX, lastY)
    lastV = lin_grid_sample(v, lastX, lastY)

    x = x - (timestep*((2.0/9.0)*firstU + (3.0/9.0)*midU + (4.0/9.0)*lastU))
    y = y - (timestep*((2.0/9.0)*firstV + (3.0/9.0)*midV + (4.0/9.0)*lastV))

    return wp.vec2(x, y)


@wp.kernel
def advect(d: Fluid,
           u: Fluid,
           v: Fluid,
           timestep: float):
    
    i, j = wp.tid()

    ux = float(i) + u.ox
    uy = float(j) + u.oy

    vx = float(i) + v.ox
    vy = float(j) + v.oy

    dx = float(i) + d.ox
    dy = float(j) + d.oy

    _dxy = rungeKutta3(u, v, dx, dy, timestep)
    _uxy = rungeKutta3(u, v, ux, uy, timestep)
    _vxy = rungeKutta3(u, v, vx, vy, timestep)

    d.dst[i,j] = cubic_grid_sample(d, _dxy[0], _dxy[1])
    u.dst[i,j] = cubic_grid_sample(u, _uxy[0], _uxy[1])
    v.dst[i,j] = cubic_grid_sample(v, _vxy[0], _vxy[1])


@wp.kernel
def buildRhs(r: wp.array2d(dtype=float), u: wp.array2d(dtype=float), v: wp.array2d(dtype=float)):
    scale = 1.0/hx
    i,j = wp.tid()
    r[i,j] = -scale*(at(u, i+1, j) - at(u, i, j) +
                            at(v, i, j+1) - at(v, i, j))

@wp.kernel
def project(r: wp.array2d(dtype=float), p_src: wp.array2d(dtype=float), p_dst: wp.array2d(dtype=float), timestep: float):

    scale = timestep / (density*hx*hx)

    i,j = wp.tid()

    diag = float(0.0)
    offDiag = float(0.0)

    if i > 0:
        diag    = diag + scale
        offDiag = offDiag - scale * at(p_src, i-1, j)
    if j > 0:
        diag    = diag + scale
        offDiag = offDiag - scale * at(p_src, i, j-1)
    if i < grid_width - 1:
        diag    = diag + scale
        offDiag = offDiag - scale * at(p_src, i+1, j)
    if j < grid_height - 1:
        diag    = diag + scale
        offDiag = offDiag - scale * at(p_src, i, j+1)
        
    newP = (at(r, i, j) - offDiag)/diag
    p_dst[i,j] = newP


@wp.kernel
def applyPressure(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), p: wp.array2d(dtype=float), timestep: float):

    scale = (timestep/1000.0)/(density*hx)
    i,j = wp.tid()

    if i == 0 or i == grid_width-1:
        return
    if j == 0 or j == grid_height-1:
        return
    

    f_p = wp.vec2(p[i+1, j] - p[i-1, j],
                  p[i, j+1] - p[i, j-1])*scale
    u[i,j] = u[i,j] - f_p[0]
    v[i,j] = v[i,j] - f_p[1]


class FluidSolver:

    def __init__(self):

        self.sim_substeps = 4
        self.sim_dt = 0.05
        self.sim_time = 0.0
        self.iterations = 600

        self.device = wp.get_device()

        self.u = Fluid()
        self.u.w = grid_width.val+1
        self.u.h = grid_height.val
        self.u.ox = 0.0
        self.u.oy = 0.5
        self.u.src = wp.zeros((self.u.w, self.u.h), dtype=float)
        self.u.dst = wp.zeros((self.u.w, self.u.h), dtype=float)

        self.v = Fluid()
        self.v.w = grid_width.val
        self.v.h = grid_height.val+1
        self.v.ox = 0.5
        self.v.oy = 0.0
        self.v.src = wp.zeros((self.v.w, self.v.h), dtype=float)
        self.v.dst = wp.zeros((self.v.w, self.v.h), dtype=float)

        self.d = Fluid()
        self.d.w = grid_width.val
        self.d.h = grid_height.val
        self.d.ox = 0.5
        self.d.oy = 0.5
        self.d.src = wp.zeros((self.d.w, self.d.h), dtype=float)
        self.d.dst = wp.zeros((self.d.w, self.d.h), dtype=float)

        shape = (grid_width.val, grid_height.val)
        self.r = wp.zeros(shape, dtype=float)
        self.p_src = wp.zeros(shape, dtype=float)
        self.p_dst = wp.zeros(shape, dtype=float)

        # capture pressure solve as a CUDA graph
        if self.device.is_cuda:
            wp.capture_begin()
            self.solve()
            self.graph = wp.capture_end()


    def solve(self):

        for j in range(self.iterations):
            wp.launch(project, dim=(grid_width.val, grid_height.val), inputs=[self.r, self.p_src, self.p_dst, self.sim_dt])
            (self.p_src, self.p_dst) = (self.p_dst, self.p_src)



    def render(self, img, i):

        for i in range(self.sim_substeps):
        
            shape = (grid_width.val, grid_height.val)
            dt = self.sim_dt

            # update emitters
            wp.launch(addInflow, dim=shape, inputs=[self.d.src, self.u.src, self.v.src, 1.0, 0.0, 3.0])

            wp.launch(buildRhs, dim=shape, inputs=[self.r, self.u.src, self.v.src])

            if self.device.is_cuda:
                wp.capture_launch(self.graph)
            else:
                self.solve()  
            
            wp.launch(applyPressure, dim=shape, inputs=[self.u.dst, self.v.dst, self.p_src, dt])

            (self.u.src, self.u.dst) = (self.u.dst, self.u.src)
            (self.v.src, self.v.dst) = (self.v.dst, self.v.src)

            # advect
            wp.launch(advect, dim=shape, inputs=[self.d, self.u, self.v, dt])

            (self.d.src, self.d.dst) = (self.d.dst, self.d.src)
            (self.u.src, self.u.dst) = (self.u.dst, self.u.src)
            (self.v.src, self.v.dst) = (self.v.dst, self.v.src)

            self.sim_time += dt

        img.set_array(self.d.src.numpy())

        return img,
        

if __name__ == '__main__':

    chapter2 = FluidSolver()

    fig = plt.figure()

    img = plt.imshow(chapter2.d.src.numpy(), origin="lower", animated=True, interpolation="antialiased")
    img.set_norm(matplotlib.colors.Normalize(0.0, 1.0))
    seq = anim.FuncAnimation(fig, lambda i: chapter2.render(img, i), frames=10000, blit=True, interval=8, repeat=False)
    # seq.save('chapter2.gif') 
    
    plt.show()
