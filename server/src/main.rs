use fosra::{Focus, Message, Role};

fn main() {
    let msg = Message {
        role: Role::User,
        content: "Hello from fosra-server!".to_string(),
    };
    println!("Hello, world! {:?}", msg);
}
