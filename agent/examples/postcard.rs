fn main() {
    use core::ops::Deref;
    use postcard::{from_bytes, to_allocvec};
    use serde::{Deserialize, Serialize};
    extern crate alloc;
    use alloc::vec::Vec;

    #[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
    struct RefStruct<'a> {
        bytes: &'a [u8],
        str_s: &'a str,
    }
    let message = "hElLo";
    let bytes = [0x01, 0x10, 0x02, 0x20];
    let output: Vec<u8> = to_allocvec(&RefStruct {
        bytes: &bytes,
        str_s: message,
    })
    .unwrap();

    dbg!("{}", &output);

    assert_eq!(
        &[
            0x04, 0x01, 0x10, 0x02, 0x20, 0x05, b'h', b'E', b'l', b'L', b'o',
        ],
        output.deref()
    );

    let out: RefStruct = from_bytes(output.deref()).unwrap();
    dbg!("{}", &out);
    assert_eq!(
        out,
        RefStruct {
            bytes: &bytes,
            str_s: message,
        }
    );
}
