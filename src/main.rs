use num::{
    traits::{
        FloatConst
    }
};
use scorus::{
    healpix::{
        interp::{
            get_interpol_ring
        }
    }
    , coordinates::{
        SphCoord
    }
};

fn main() {
    let ntheta=100;
    let nphi=100;
    let theta_min=0.0;
    let theta_max=f64::PI();
    let phi_min=0.0;
    let phi_max=2.0*f64::PI();
    let theta_list=(0..ntheta).map(|i| (theta_max-theta_min)/ntheta as f64*i as f64+theta_min).collect::<Vec<_>>();
    let phi_list=(0..nphi).map(|i| (phi_max-phi_min)/nphi as f64*i as f64+phi_min).collect::<Vec<_>>();
    for &theta in &theta_list{
        for &phi in &phi_list{
            let ptg=SphCoord::<f64>::new(theta, phi);
            let (p,w)=get_interpol_ring(16, ptg);
            
            let mut pw:Vec<_>=p.iter().cloned().zip(w.iter().cloned()).collect();
            pw.sort_by_key(|x| x.0);
            let p:Vec<_>=pw.iter().map(|x| x.0).collect();
            let w:Vec<_>=pw.iter().map(|x| x.1).collect();
            for i in p{
                print!("{} ", i);
            }
            for i in w{
                print!("{} ", i);
            }
            println!();
            
        }
    }
}
