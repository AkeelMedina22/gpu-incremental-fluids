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
    

@wp.kernel
def addInflow(d_src: wp.array2d(dtype=float),
              u_src: wp.array2d(dtype=float),
              v_src: wp.array2d(dtype=float), d: float, u: float, v: float):

    i, j = wp.tid()

    if i < 220 and i > 180 and j < 220 and j > 180:
        if d_src[i,j] < d:
            d_src[i,j] = d
        if u_src[i,j] < u:
            u_src[i,j] = u
        if v_src[i,j] < v:
            v_src[i,j] = v


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
def euler(u: Fluid, v: Fluid, x: float, y: float, timestep: float):
    uVel = lin_grid_sample(u, x, y)/hx
    vVel = lin_grid_sample(v, x, y)/hx
    
    x = x - uVel*timestep
    y = y - vVel*timestep

    return wp.vec2(x, y)


@wp.kernel
def advect(d: Fluid,
           u: Fluid,
           v: Fluid,
           timestep: float):

    i,j = wp.tid()
    
    ux = float(i) + u.ox
    uy = float(j) + u.oy

    vx = float(i) + v.ox
    vy = float(j) + v.oy

    dx = float(i) + d.ox
    dy = float(j) + d.oy

    _dxy = euler(u, v, dx, dy, timestep)
    _uxy = euler(u, v, ux, uy, timestep)
    _vxy = euler(u, v, vx, vy, timestep)

    d.dst[i,j] = lin_grid_sample(d, _dxy[0], _dxy[1])
    u.dst[i,j] = lin_grid_sample(u, _uxy[0], _uxy[1])
    v.dst[i,j] = lin_grid_sample(v, _vxy[0], _vxy[1])


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

    i,j = wp.tid()

    if i == 0 or i == grid_width-1:
        return
    if j == 0 or j == grid_height-1:
        return
    

    scale = (timestep/1000.0)/(density*hx)
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

    chapter1 = FluidSolver()

    fig = plt.figure()

    img = plt.imshow(chapter1.d.src.numpy(), origin="lower", animated=True, interpolation="antialiased")
    img.set_norm(matplotlib.colors.Normalize(0.0, 1.0))
    seq = anim.FuncAnimation(fig, lambda i: chapter1.render(img, i), frames=100000, blit=True, interval=8, repeat=False)
    # seq.save('chapter1.gif') 
    
    plt.show()