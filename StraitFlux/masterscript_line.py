import xarray as xa
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("skipping matplotlib")
import sys
from functools import partial

try:
    from dask.diagnostics import ProgressBar
except ImportError:
    print("skipping dask import")

import StraitFlux.preprocessing as prepro
import StraitFlux.functions as func
from StraitFlux.indices import check_availability_indices, prepare_indices


def transports(
    product,
    strait,
    model,
    time_start,
    time_end,
    file_u,
    file_v,
    file_t,
    file_z,
    mesh_dxv=0,
    mesh_dyu=0,
    coords=0,
    set_latlon=False,
    lon_p=0,
    lat_p=0,
    file_s="",
    file_sic="",
    file_sit="",
    Arakawa="",
    rho=1026,
    cp=3996,
    Tref=0,
    path_save="",
    path_indices="",
    path_mesh="",
    showplots = False,
    saveplots = True,
    savegrid = True,
    save_indices = True,
    saving=True,
):
    """Calculation of Transports using line integration

    INPUT Parameters:
    product (str): volume, heat, salt or ice
    strait (str): desired oceanic strait, either pre-defined from indices file or new
    model (str): desired CMIP6 model or reanalysis
    time_start (str or int): starting year
    time_end (str or int): ending year
    file_u (str OR ): path + filename(s) of u field(s); use ice velocities (ui) for ice transports; (multiple files possible, use *; must be possible to combine files over time coordinate)
    file_v (str): path + filename(s) of v field(s); use ice velocities (vi) for ice transports; (multiple files possible, use *)
    file_t (str): path + filename(s) of temperature field(s); (multiple files possible, use *)
    file_z (str): path + filename(s) of cell thickness field(s); (multiple files possible, use *)

    OPTIONAL:
    mesh_dxu/mesh_dyv (array): arrays containing the exact grid cell dimensions at northern and eastern grid cell faces of u and v cells (dxv and dyu); if not supplied will be calculated
    coords (tuple): coordinates for strait, if not pre-defined: (latitude_start,longitude_start,latitude_end,longitude_end)
    set_latlon: set True if you wish to pass arrays of latitudes and longitudes
    lon (array): longitude coordinates for strait, if not pre-defined. (range -180 to 180; same length as lat needed!)
    lat (array): latitude coordinates for strait, if not pre-defined. (range -90 to 90; same length as lon needed!)
    file_s (str): only needed for salinity transports; path + filename(s) of salinity field(s); (multiple files possible, use *)
    file_sic (str): only needed for ice transports; path + filename(s) of sea ice concentration field(s); (multiple files possible, use *)
    file_sit (str): only needed for ice transports; path + filename(s) of sea ice thickness field(s); (multiple files possible, use *)
    Arakawa (str): Arakawa-A, Arakawa-B or Arakawa-C; only needed if automatic check fails
    rho (int or array): default = 1026 kg/m3
    cp (int or array): default = 3996 J/(kgK)
    Tref (int or array): default = 0°C
    path_save (str): path to save transport data
    path_indices (str): path to save indices data
    path_mesh (str): path to save mesh data


    RETURNS:
    volume, heat, salt or ice transports through specified strait for specified model

    """

    partial_func = partial(prepro._preprocess1)

    print("read and load files for indices, grid and mesh")

    # load files for the indices, depending on if the file is provided as an xarray
    # file or as a string, read the file

    if type(file_t) == str:
        ti = xa.open_mfdataset(file_t, preprocess=partial_func).isel(time=0)
    elif type(file_t) == xa.core.dataset.Dataset:
        ti = partial_func(file_t).isel(time=0)
    else:
        print("please provide either a string or an array dataset")

    if type(file_u) == str:
        ui = xa.open_mfdataset(file_u, preprocess=partial_func).isel(time=0)
    elif type(file_u) == xa.core.dataset.Dataset:
        ui = partial_func(file_u).isel(time=0)
    else:
        print("please provide either a string or an array dataset")

    if type(file_v) == str:
        vi = xa.open_mfdataset(file_v, preprocess=partial_func).isel(time=0)
    elif type(file_v) == xa.core.dataset.Dataset:
        vi = partial_func(file_v).isel(time=0)
    else:
        print("please provide either a string or an array dataset")
    
    # if the files have been loaded previously, the progressbar will not show
    # because the process is too quick
    try:
        with ProgressBar():
            ti = ti.load()
            ui = ui.load()
            vi = vi.load()
    except NameError:
        ti = ti.load()
        ui = ui.load()
        vi = vi.load()

    # to save memory, if the indices are calculated before already, use that
    try:
        indices = xa.open_dataset(path_indices + model + "_" + strait + "_indices.nc")
    
    # if the indices haven't been calculated before, use the datasets read above
    except OSError:
        print("calc indices")
        
        # check_availability indices is a funcction from the indices.py script
        # indices returns an xarray dataset with two dimensions, and it prints
        # a number from the indices.check_res() function
        indices, line = check_availability_indices(
            ti, strait, model, coords, lon_p, lat_p, set_latlon
        )

        # prepare_indices prepares the points when ugrid or vgrid have to be choosen.
        # out_u/ out_v: np.array with indices where ugrid/ vgrid needs to be selected
        # out_u_vz: np.array with signs for the indices on ugrid
        out_u, out_v, out_u_vz = prepare_indices(indices)

        # selecting the indices that are not 0
        i2 = indices.indices.where(indices.indices != 0)
        
        # create a plot with the indices, showing and saving is optional
        try:
            plt.pcolormesh((ti.thetao / ti.thetao), cmap="tab20c")
            plt.scatter(i2[:, 2], i2[:, 3], color="tab:red", s=0.1, marker="x")
            plt.scatter(i2[:, 0], i2[:, 1], color="tab:red", s=0.1, marker="x")
            plt.title(model + "_" + strait, fontsize=14)
            plt.ylabel("y", fontsize=14)
            plt.xlabel("x", fontsize=14)
            if showplots:
                plt.show()
            if saveplots:
                plt.savefig(path_save + strait + "_" + model + "_indices.png")
            plt.close()
        except NameError:
            print("skipping Plot")
               
        func.check_indices(
            indices, out_u, out_v, ti, ui, vi, strait, model, path_indices
            )
        if save_indices:
            print('indices are saved to netcdf')
            indices.to_netcdf(path_save + model + "_" + strait + "_indices.nc")
        else:
            print('indices are not saved')

    out_u, out_v, out_u_vz = prepare_indices(indices)
    
    # check the grid
    if Arakawa in ["Arakawa-A", "Arakawa-B", "Arakawa-C"]:
        grid = Arakawa
        print('grid is ', Arakawa)
    elif Arakawa == "":

        # try if a grif file is provided or produced in a previous run of the function
        try:
            file = open(path_mesh + model + "grid.txt", "r")
            grid = file.read()
        
        # if no grid file is provided, the line above will throw an error
        except OSError:
            # determines the grid based on the latitudes and longitude
            grid = func.check_Arakawa(ui, vi, ti, model)
            
        if savegrid:
            with open(path_save + model + "grid.txt", "w") as f:
                f.write(grid)
            
    else:
        print("grid not known")
        sys.exit()

    # return min and max x and y from u and v
    min_x = np.nanmin(
        (min(out_u[:, 0], default=np.nan), min(out_v[:, 0], default=np.nan))
    )
    max_x = np.nanmax(
        (max(out_u[:, 0], default=np.nan), max(out_v[:, 0], default=np.nan))
    )
    min_y = np.nanmin(
        (min(out_u[:, 1], default=np.nan), min(out_v[:, 1], default=np.nan))
    )
    max_y = np.nanmax(
        (max(out_u[:, 1], default=np.nan), max(out_v[:, 1], default=np.nan))
    )

    if min_x == -1:
        min_x = 0
        max_x = max_x + 1

    # if the function was run before, the mesh for u and v will already have been
    # calculated and saved
    try:
        mu = xa.open_dataset(path_mesh + "mesh_dyu_" + model + ".nc")
        mv = xa.open_dataset(path_mesh + "mesh_dxv_" + model + ".nc")

    # if the function was not run before, either calculate the dxu/ dyu meshes
    # or use the ones provided.
    except FileNotFoundError:
        if mesh_dxv != 0:
            mv = mesh_dxv.to_dataset(name="dxv")
            mv.to_netcdf(
                path_save + "mesh_dxv_" + model + ".nc"
            )
            mu = mesh_dyu.to_dataset(name="dyu")
            mu.to_netcdf(
                path_save + "mesh_dyu_" + model + ".nc"
            )
        else:
            print("calc horizontal meshes")
            mu, mv = prepro.calc_dxdy(model, ui, vi, path_mesh)

    print("read t, u and v fields")
    # for the actual u and v fielts, t, u and v are used
    partial_func = partial(
        prepro._preprocess2,
        lon_bnds=(int(min_x) - 1, int(max_x) + 1),
        lat_bnds=(int(min_y) - 1, int(max_y) + 1),
    )

    timeslice = slice(str(time_start), str(time_end))

    if type(file_t) == str:
        t = xa.open_mfdataset(file_t, preprocess=partial_func2, chunks={"time": 1})
    elif type(file_t) == xa.core.dataset.Dataset:
        t = partial_func2(file_t).chunk({"time": 1})
    else:
        print("please provide either a string or an array dataset")
    if type(file_u) == str:
        u = xa.open_mfdataset(file_u, preprocess=partial_func2, chunks={"time": 1})
    elif type(file_u) == xa.core.dataset.Dataset:
        u = partial_func2(file_u).chunk({"time": 1})
    else:
        print("please provide either a string or an array dataset")
    if type(file_v) == str:
        v = xa.open_mfdataset(file_v, preprocess=partial_func2, chunks={"time": 1})
    elif type(file_v) == xa.core.dataset.Dataset:
        v = partial_func2(file_v).chunk({"time": 1})
    else:
        print("please provide either a string or an array dataset")

    if "time" in t.dims and t.sizes["time"] > 1:
        t = t.sel(time=timeslice)
        u = u.sel(time=timeslice)
        v = v.sel(time=timeslice)
    elif "time" not in t.dims:
        t = t.expand_dims(dim={"time": 1})
        u = u.expand_dims(dim={"time": 1})
        v = v.expand_dims(dim={"time": 1})

    # open the file with the cell thickness. According to the paper, there should be a 
    # function calc_dydz, but it is not in the package
    try:
        if type(file_z) == str:
            deltaz = xa.open_mfdataset(
                file_z, preprocess=partial_func2, chunks={"time": 1}
            )[["thkcello"]]
        elif type(file_z) == xa.core.dataset.Dataset:
            deltaz = partial_func2(file_z).chunk({"time": 1})[["thkcello"]]
    except KeyError:
        print("here you need to do something about using deptho instead of thkcello")
        sys.exit()

    if "time" in deltaz.dims:
        deltaz = deltaz.sel(time=timeslice)
    mu = mu.sel(
        x=slice(int(min_x) - 1, int(max_x) + 1), y=slice(int(min_y) - 1, int(max_y) + 1)
    ).load()
    mv = mv.sel(
        x=slice(int(min_x) - 1, int(max_x) + 1), y=slice(int(min_y) - 1, int(max_y) + 1)
    ).load()

    print("load t, u and v fields")
    # if the files have already been read before, the progressbar will not show
    # because the progress is too fast
    try:
        with ProgressBar():
            t = t.load()
            u = u.load()
            v = v.load()
            deltaz = deltaz.load()
    except NameError:
        t = t.load()
        u = u.load()
        v = v.load()
        deltaz = deltaz.load()

    # with dzu3 and dzv3
    dzu3, dzv3 = func.calc_dz_faces(deltaz, grid, model, path_mesh)
        
    sign_v = []
    
    indi = indices.indices[:, 2][indices.indices[:, 3] != 0]
    for ind in range(len(indi) - 1):
        if indi[ind] < indi[ind + 1]:
            sign_v = np.append(sign_v, 1)
        elif indi[ind] >= indi[ind + 1]:
            if indi[ind + 1] in [1, 0, -1]:
                sign_v = np.append(sign_v, 1)
            else:
                sign_v = np.append(sign_v, -1)
    try:
        sign_v = np.append(sign_v, sign_v[-1])
    except IndexError:
        pass

    # read salinity and 'substance' datasets
    if product == "salt":
        if type(file_s) == str:
            Sdata = xa.open_mfdataset(
                file_s, preprocess=partial_func2, chunks={"time": 1}
            ).sel(time=timeslice)
        elif type(file_s) == xa.core.dataset.Dataset:
            Sdata = partial_func2(file_s).chunk({"time": 1}).sel(time=timeslice)

    if product == "substance":
        if type(file_substance) == str:
            Substancedata = xa.open_mfdataset(
                file_substance, preprocess=partial_func2, chunks={"time": 1}
            ).sel(time=timeslice)
        elif type(file_substance) == xa.core.dataset.Dataset:
            Substancedata = (
                partial_func2(file_substance).chunk({"time": 1}).sel(time=timeslice)
            )
    
    # start the transport calculations
    print(" ...calculating transport")
    udata = u.uo
    vdata = v.vo
    Tdata = t

    # transform the product
    if product in ["volume", "heat", "salt", "substance"]:
        udata, vdata2, dzu3, dzv3, mu2, mv2 = func.transform_Arakawa(
            grid, mu, mv, deltaz, dzu3, dzv3, udata, vdata
        )

    # create a data array with the volumes for u and v
    # depending on the native grid (Arakawa), and the mesh, mu2 and mv2 are
    # calculated in the func.transform_Arakawa function
    if product == "volume":
        print("calc u: the x-ward volumetric transport rate in m3 s-1")
        udata = udata * mu2.dyu.values * dzu3.values
        print("calc v: the y-ward volumetric transport rate in m3 s-1")
        vdata = vdata * mv2.dxv.values * dzv3.values
        unit = 'm3 s-1'
        endunit = 'm3 s-1'

    if product == "heat":
        print("rolling T")
        Tudata = func.interp_TS(Tdata.thetao, "x")
        Tvdata = func.interp_TS(Tdata.thetao, "y")
        print("calc u: the x-ward transport rate of temperature in degC m3 s-1")
        udata = udata * mu2.dyu.values * dzu3.values * (Tudata.values - Tref)
        print("calc v: the y-ward transport rate of temperature in degC m3 s-1")
        vdata = vdata * mv2.dxv.values * dzv3.values * (Tvdata.values - Tref)
        # note: to calculate heat transport, this needs to be multiplied by the 
        # specific heat capacity and the density.
        unit = 'degC m3 s-1'
        endunit = 'degC s-1'

    if product == "salt":
        print("rolling S")
        Sudata = func.interp_TS(Sdata.so, "x")
        Svdata = func.interp_TS(Sdata.so, "y")
        print("calc u: the x-ward salinity transport rate in PSU m3 s-1")
        udata = udata * mu2.dyu.values * dzu3.values * Sudata.values
        print("calc v: the y-ward salinity transport rate in PSU m3 s-1")
        vdata = vdata * mv2.dxv.values * dzv3.values * Svdata.values
        unit = 'PSU m3 s-1'
        endunit = 'PSU s-1'

    if product == "substance":
        print("rolling Substance")
        Substanceudata = func.interp_TS(Substancedata[substance_name], "x")
        Substancevdata = func.interp_TS(Substancedata[substance_name], "y")
        print("calc u: the x-ward ", Substancedata[substance_name].long_name, " transport rate in ", 
              Substancedata[substance_name].units, " m3 s-1")
        udata = udata * mu2.dyu.values * dzu3.values * Substanceudata.values
        print("calc u: the y-ward ", Substancedata[substance_name].long_name, " transport rate in ", 
              Substancedata[substance_name].units, " m3 s-1")
        vdata = vdata * mv2.dxv.values * dzv3.values * Substancevdata.values
        unit = str(Substancedata[substance_name].units + " m3 s-1")
        endunit = str(Substancedata[substance_name].units + " s-1")

    if product == "ice":
        print("calc u")
        udata = udata * mu.dyu.values * sit.sithick.values * sic.siconc.values
        print("calc v")
        vdata = vdata * mv.dxv.values * sit.sithick.values * sic.siconc.values

    # replace all nans by 0
    udata = udata.fillna(0.0)
    vdata = vdata.fillna(0.0)

    print("calc line")
    # here, we sum over depth
    if product in ["volume", "heat", "salt"]:
        udata = udata.sum(dim="lev")
        vdata = vdata.sum(dim="lev")

    # create xarray datasets for datau and datav with dimensions time, x and y
    datau = xa.Dataset(
        {"inte": (("time", "y", "x"), udata.data)},
        coords={
            "time": ("time", udata.time.data),
            "x": ("x", udata.x.data),
            "y": ("y", udata.y.data)
        }
    )
    datav = xa.Dataset(
        {"inte": (("time", "y", "x"), vdata.data)},
        coords={
            "time": ("time", vdata.time.data),
            "x": ("x", vdata.x.data),
            "y": ("y", vdata.y.data),
        }
    )

    # create empty arrays for pointsu and pointsv
    pointsu = np.zeros(datau.inte.shape)
    pointsv = np.zeros(datav.inte.shape)

    data1u = np.array(datau.inte)
    data1v = np.array(datav.inte)

    # modify pointsu based on out_u_vz conditions
    for l in range(len(out_u)):
        index_y = int(out_u_vz[l, 1] - min_y + 1)
        index_x = int(out_u_vz[l, 0] - min_x + 1)

        if out_u_vz[l][2] == -1:
            pointsu[:, index_y, index_x] = data1u[:, index_y, index_x] * -1
        else:
            pointsu[:, index_y, index_x] = data1u[:, index_y, index_x]

    # indi1 and indi2 are the indices of the cells of interest
    indi1 = indices.indices[:, 2][indices.indices[:, 3] != 0]
    indi2 = indices.indices[:, 3][indices.indices[:, 3] != 0]

    # modify pointsv based on indices
    for m in range(len(indi1) - 1):
        index_y = int(indi2[m] - min_y + 1)
        index_x = int(indi1[m] - min_x + 1)
        pointsv[:, index_y, index_x] = data1v[:, index_y, index_x] * sign_v[m]

    # combine and process the data
    vp = xa.Dataset({"v": (("time", "y", "x"), pointsv)}, coords=datav.coords)
    up = xa.Dataset({"u": (("time", "y", "x"), pointsu)}, coords=datau.coords)
    
    combined_l = datau.copy()
    ges_l["inte"] = vp["v"] + up["u"]
    
    # calculate heat transport H = Q * rho * cp * deltat (deltat calculated in udata)
    if product == "heat":
        combined_l["inte"] = combined_l["inte"] * rho * cp
    elif product == "salt":
        combined_l["inte"] = combined_l["inte"] * rho
    
    # integrate transport over x and y to get a total transport over the transect,
    # a split in in- and outflow can be made here
    trans = combined_l.sum(dim=["x", "y"]).rename(({'inte':'tot_' + product + 'flux'}))

    trans = trans.assign_attrs(
         transect = strait,
         model = model,
         variable_id = str("tot_" + product + "_flux"))
    trans[str("tot_" + product + "_flux")] = trans[str("tot_" + product + "_flux")].assign_attrs(
         standard_name = str('Transport_rate_of_' + product),
         long_name = str('Total ' + product + ' flux'), 
         units = endunit,
         comment = 'Transport rate calculated using the StraitFlux transports() function',
         ) 
    
    if savetransport:
        trans.to_netcdf(path_save + '_'.join([strait, product, model, years]) + ".nc")

    return trans
