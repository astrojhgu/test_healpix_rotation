use scorus::{
    healpix::{
        pix::{
            pix2ang_ring
            , pix2vec_ring
        }
        , interp::{
            natural_interp_ring
        }
        , utils::{
            nside2npix
            , npix2nside
        }
        , rotation::{
            get_euler_matrix
            , get_rotation_matrix
            , rotate_ring
            , rotate_ring_pol
        }
    }
    , coordinates::{
        SphCoord
        , Vec3d
    }
};

use healpix_fits::{
    write_map
};

fn main(){
    let mat=get_rotation_matrix(45_f64.to_radians(), 45_f64.to_radians(), 10_f64.to_radians());
    //let mat=get_rotation_matrix(0_f64.to_radians(), 0_f64.to_radians(), 0_f64.to_radians());
    eprintln!("{:?}", mat);

    let nside=256;
    let npix=nside2npix(nside);
    //let data:Vec<_>=(0..npix).map(|i| (i as f64/1024.0*2.0*3.14159).sin()).collect();
    let pt0=SphCoord::new(45_f64.to_radians(), 0.0_f64.to_radians());
    let t0=(0..npix).map(|i| (-pix2ang_ring::<f64>(nside, i).angle_between(pt0).powi(2)).exp()).collect::<Vec<_>>();
    let pq0=SphCoord::new(90_f64.to_radians(), 0.0_f64.to_radians());
    let q0=(0..npix).map(|i| (-pix2ang_ring::<f64>(nside, i).angle_between(pq0).powi(2)).exp()).collect::<Vec<_>>();
    let pu0=SphCoord::new(135_f64.to_radians(), 0.0_f64.to_radians());
    let u0=(0..npix).map(|i| (-pix2ang_ring::<f64>(nside, i).angle_between(pu0).powi(2)).exp()).collect::<Vec<_>>();

    write_map("t0.fits" , &[&t0], false, true);
    write_map("q0.fits" , &[&q0], false, true);
    write_map("u0.fits" , &[&u0], false, true);
    
    let (t1,q1,u1)=rotate_ring_pol(&t0, &q0, &u0, &mat);
    write_map("t1.fits" , &[&t1], false, true);
    write_map("q1.fits" , &[&q1], false, true);
    write_map("u1.fits" , &[&u1], false, true);
}
