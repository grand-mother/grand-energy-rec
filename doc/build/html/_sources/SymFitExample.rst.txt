Symmetric Fit Example
*********************

>>> from modERec import EnergyRec
>>> import numpy as np

:math:`a_{ratio}` trainning
---------------------------

>>> def quantities(simulation):
>>> rec = EnergyRec(simulation)
>>> 
>>> weight = np.array([ant.wEarlyLate for ant in rec.antenna])
>>> r_proj = np.array([ant.r_proj for ant in rec.antenna])
>>> fluence_geo = np.array([ant.fluence_geo for ant in rec.antenna])/(weight**2)
>>> fluence_ce = np.array([ant.fluence_ce for ant in rec.antenna])/(weight**2)
>>> 
>>> n_ant = len(rec.GRANDshower.fields)
>>> alpha = np.full(n_ant,np.arccos(np.dot(rec.shower.ev,rec.shower.eB)))
>>> distance = np.linalg.norm((r_proj - rec.shower.r_Core_proj)[:,0:2],axis=1)*weight
>>> d_Xmax = np.full(n_ant,rec.shower.d_Xmax)
>>> rho_Xmax = np.full(n_ant,SymFit.rho(d_Xmax[0],-rec.shower.ev))
>>> 
>>> return fluence_geo, fluence_ce, alpha, distance, d_Xmax, rho_Xmax

>>> fluence_geo = np.array([])
>>> fluence_ce = np.array([])
>>> alpha = np.array([])
>>> distance = np.array([])
>>> d_Xmax = np.array([])
>>> rho_Xmax = np.array([])
>>> 
>>> for i in range(100001,10009):
>>>     simulation = "sim" + str(i) + ".hdf5"
>>> 
>>>     f_geo, f_ce, a, d, d_max, rho_max = quantities(simulation)
>>>     fluence_geo = np.append(fluence_geo,f_geo)
>>>     fluence_ce = np.append(fluence_ce,f_ce)
>>>     alpha = np.append(alpha,a)
>>>     distance = np.append(distance,d)
>>>     d_Xmax = np.append(d_Xmax,d_max)
>>>     rho_Xmax = np.append(rho_Xmax,rho_max)

>>> bestfit = SymFit.a_ratio_fit(fluence_geo, fluence_ce,alpha, distance, d_Xmax, rho_Xmax)

:math:`S_{radio}` trainning
---------------------------

>>> ldf_par_arr = np.empty(shape=(0, 4))
>>> alpha_arr = np.array([])
>>> rho_Xmax_arr = np.array([])
>>> E_arr = np.array([])
>>> 
>>> for i in range(100001,100008):
>>>     simulation = "sim" + str(i) + ".hdf5"
>>>     rec = EnergyRec(simulation)
>>>  
>>>     r_proj = np.array([ant.r_proj for ant in rec.antenna])
>>>     distance = np.linalg.norm((r_proj - rec.shower.r_Core_proj)[:,0:2],axis=1)
>>>     
>>>     fluence_par = rec.Eval_par_fluences(bestfit)
>>>     sel = np.where(fluence_par>0)
>>>     
>>>     ldf_par = SymFit.SymLDF_fit(distance[sel],fluence_par[sel])
>>>     d_Xmax = rec.shower.d_Xmax
>>>     rho_Xmax = SymFit.rho(d_Xmax,-rec1.shower.ev)
>>> 
>>>     ldf_par_arr = np.vstack((ldf_par_arr,[ldf_par]))
>>>     alpha_arr = np.append(alpha_arr,[np.arccos(np.dot(rec1.shower.ev,rec1.shower.eB))])
>>>     rho_Xmax_arr = np.append(rho_Xmax_arr,[rho_Xmax])
>>>     E_arr = np.append(E_arr,[rec1.GRANDshower.energy.to("EeV").value])

Perform the joint Fit
---------------------

>>> joint_bestfit = SymFit.joint_S_fit(ldf_par_arr,alpha_arr,rho_Xmax_arr,E_arr)

:math:`S_{radio}\,\times` energy relation
-----------------------------------------

>>> import matplotlib.pyplot as plt
>>> 
>>> S_geo = np.zeros(len(ldf_par_arr))
>>> S_mod = np.zeros(len(ldf_par_arr))
>>> 
>>> for i in range(len(ldf_par_arr)):
>>>     S_geo[i] = SymFit.Sradio_geo(joint_bestfit[0:2],ldf_par_arr[i],alpha_arr[i],rho_Xmax_arr[i])
>>>     S_mod[i] = SymFit.Sradio_mod(joint_bestfit[2:4],E_arr[i])
>>> 
>>> plt.figure(figsize=(14,5))
>>> plt.subplot(121)
>>> plt.plot(E_arr*1.e18,S_geo,"*",label="MC")
>>> 
>>> fit_label = "fit: "+ "{:.2e}".format(joint_bestfit[2]) + " x pow(E,"+ "{:.2f}".format(joint_bestfit[3]) +")"
>>> fit_x = [np.min(E_arr*1.e18),np.max(E_arr*1.e18)]
>>> fit_y = [SymFit.Sradio_mod(joint_bestfit[2:4],np.min(E_arr)), SymFit.Sradio_mod(joint_bestfit[2:4],np.max(E_arr))]
>>> plt.plot(fit_x,fit_y,label = fit_label)
>>> plt.legend()
>>> plt.xlabel(r"E$_{MC}$ in eV")
>>> plt.ylabel(r"S$_{radio}$ in GeV")
>>> plt.gca().set_yscale('log')
>>> plt.gca().set_xscale('log')
>>> 
>>> plt.subplot(122)
>>> 
>>> erec = pow((S_geo/joint_bestfit[2]),1./joint_bestfit[3])*10./E_arr
>>> e_label = r"E$_{radio}$/E_${MC}$"
>>> stat = "mean = " +"{:.2f}".format(np.mean(erec)) + r" ; $\sigma$ = " + "{:.2f}".format(np.std(erec))
>>> plt.hist(erec,label=stat)
>>> plt.xlabel(e_label)
>>> plt.xlim(0.8,1.2)
>>> plt.ylabel("#")
>>> plt.legend()