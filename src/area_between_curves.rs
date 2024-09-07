use num::Float;

fn cross_2d<T>(v0_0 : T, v0_1 : T, v1_0 : T, v1_1 : T) -> T
where
T : Float
{
    (v0_0 * v1_1) - (v0_1 * v1_0)
}

pub fn is_simple_quad<T>(
    v0_0 : T, v0_1 : T,
    v1_0 : T, v1_1 : T,
    v2_0 : T, v2_1 : T,
    v3_0 : T, v3_1 : T
) -> bool
where
T : Float
{
    let mut ts : [T; 4] = [T::zero(); 4];
    ts[0] = cross_2d(v0_0, v0_1, v1_0, v1_1);
    ts[1] = cross_2d(v1_0, v1_1, v2_0, v2_1);
    ts[2] = cross_2d(v2_0, v2_1, v3_0, v3_1);
    ts[3] = cross_2d(v3_0, v3_1, v0_0, v0_1);

    let mut pos : u8 = 0;
    let mut neg : u8 = 0;
    let mut zer : u8 = 0;

    for elem in ts.iter() {
        if *elem > T::zero() {
            pos += 1;
        } else if *elem < T::zero() {
            neg += 1;
        } else {
            zer += 1;
        }
    }

    let tf = if pos < neg {neg + zer} else {pos + zer};

    tf > 2    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_simple_quad0() {
        let x = [0.0, 1.0, 1.0, 0.0];
        let y = [0.0, 1.0, 0.0, 1.0];

        let ab = [x[1]-x[0], y[1]-y[0]];
        let bc = [x[2]-x[1], y[2]-y[1]];
        let cd = [x[3]-x[2], y[3]-y[2]];
        let da = [x[0]-x[3], y[0]-y[3]];

        let quad = is_simple_quad(
            ab[0], ab[1],
            bc[0], bc[1],
            cd[0], cd[1],
            da[0], da[1],
        );
        
        assert!(!quad);
    }

    #[test]
    fn test_is_simple_quad1() {
        let x = [0.0, 0.0, 1.0, 1.0];
        let y = [0.0, 1.0, 1.0, 0.0];

        let ab = [x[1]-x[0], y[1]-y[0]];
        let bc = [x[2]-x[1], y[2]-y[1]];
        let cd = [x[3]-x[2], y[3]-y[2]];
        let da = [x[0]-x[3], y[0]-y[3]];

        let quad = is_simple_quad(
            ab[0], ab[1],
            bc[0], bc[1],
            cd[0], cd[1],
            da[0], da[1],
        );
        
        assert!(quad);
    }

    #[test]
    fn test_is_simple_quad2() {
        let x = [0.0, 1.0, 1.0, 0.0];
        let y = [0.0, 0.0, 1.0, 0.0];

        let ab = [x[1]-x[0], y[1]-y[0]];
        let bc = [x[2]-x[1], y[2]-y[1]];
        let cd = [x[3]-x[2], y[3]-y[2]];
        let da = [x[0]-x[3], y[0]-y[3]];

        let quad = is_simple_quad(
            ab[0], ab[1],
            bc[0], bc[1],
            cd[0], cd[1],
            da[0], da[1],
        );
        
        assert!(quad);
    }

    #[test]
    fn test_is_simple_quad3() {
        let x = [0.0, 1.0, 1.0, 0.0];
        let y = [0.0, 1.0, 0.0, 0.0];

        let ab = [x[1]-x[0], y[1]-y[0]];
        let bc = [x[2]-x[1], y[2]-y[1]];
        let cd = [x[3]-x[2], y[3]-y[2]];
        let da = [x[0]-x[3], y[0]-y[3]];

        let quad = is_simple_quad(
            ab[0], ab[1],
            bc[0], bc[1],
            cd[0], cd[1],
            da[0], da[1],
        );
        
        assert!(quad);
    }
}
