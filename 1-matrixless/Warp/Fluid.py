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

    return f[x,y]
    

@wp.kernel
def addInflow(d_src: wp.array2d(dtype=float),
              u_src: wp.array2d(dtype=float),
              v_src: wp.array2d(dtype=float), d: float, u: float, v: float):

    i, j = wp.tid()

    if i < 25 and i > 20 and j < 220 and j > 180:
        if d_src[i,j] < d:
            d_src[  i,j] = d
        if u_src[i,j] < u:
            u_src[i,j] = u
        if v_src[i,j] < v:
            v_src[i,j] = v


@wp.func
def lin_grid_sample(f: wp.array2d(dtype=float), x: float, y: float):

    lx = int(wp.floor(x))
    ly = int(wp.floor(y))

    tx = x-float(lx)
    ty = y-float(ly)
    
    s0 = wp.lerp(at(f, lx, ly), at(f, lx+1, ly), tx)
    s1 = wp.lerp(at(f, lx, ly+1), at(f, lx+1, ly+1), tx)

    s = wp.lerp(s0, s1, ty)
    return s


@wp.kernel
def advect(d_src: wp.array2d(dtype=float),
           d_dst: wp.array2d(dtype=float),
           u_src: wp.array2d(dtype=float),
           u_dst: wp.array2d(dtype=float),
           v_src: wp.array2d(dtype=float),
           v_dst: wp.array2d(dtype=float),
           timestep: float):

    i,j = wp.tid()
    
    x = float(i)
    y = float(j)

    uVel = lin_grid_sample(u_src, x, y)
    vVel = lin_grid_sample(v_src, x, y)

    x = x - uVel*timestep
    y = y - vVel*timestep

    d_dst[i,j] = lin_grid_sample(d_src, x, y)
    u_dst[i,j] = lin_grid_sample(u_src, x, y)
    v_dst[i,j] = lin_grid_sample(v_src, x, y)


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

    scale = timestep / (density * hx)
    i,j = wp.tid()

    u[i,    j]    = u[i,    j]    - scale*p[i,j]
    u[(i+1),j]    = u[(i+1),j]    + scale*p[i,j]
    v[i,    j]    = v[i,    j]    - scale*p[i,j]
    v[i,   (j+1)] = v[i,   (j+1)] + scale*p[i,j]


    if i == 0 or i == grid_width:
        u[i, j] = 0.0
    if j == 0 or j == grid_height:
        v[i, j] = 0.0


class FluidSolver:

    def __init__(self):

        self.sim_substeps = 1
        self.sim_dt = 0.05
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
            wp.launch(addInflow, dim=shape, inputs=[self.d_src, self.u_src, self.v_src, 1.0, 0.0, 3.0])

            wp.launch(buildRhs, dim=shape, inputs=[self.r, self.u_src, self.v_src])

            for j in range(self.iterations):
                wp.launch(project, dim=shape, inputs=[self.r, self.p_src, self.p_dst, dt])
                (self.p_src, self.p_dst) = (self.p_dst, self.p_src)

            wp.launch(applyPressure, dim=shape, inputs=[self.u_src, self.v_src, self.p_src, dt])

            # advect
            wp.launch(advect, dim=shape, inputs=[self.d_src, self.d_dst, self.u_src, self.u_dst, self.v_src, self.v_dst, dt])

            (self.d_src, self.d_dst) = (self.d_dst, self.d_src)
            (self.u_src, self.u_dst) = (self.u_dst, self.u_src)
            (self.v_src, self.v_dst) = (self.v_dst, self.v_src)

            self.sim_time += dt

        img.set_array(self.d_src.numpy())

        return img,
        

if __name__ == '__main__':

    chapter1 = FluidSolver()

    fig = plt.figure()

    img = plt.imshow(chapter1.d_src.numpy(), origin="lower", animated=True, interpolation="antialiased")
    img.set_norm(matplotlib.colors.Normalize(0.0, 1.0))
    seq = anim.FuncAnimation(fig, lambda i: chapter1.render(img, i), frames=10000, blit=True, interval=8, repeat=False)
    # seq.save('chapter1.gif') 
    
    plt.show()