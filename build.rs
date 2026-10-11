use anyhow::Result;
use vergen_git2::Git2Builder;
use vergen_lib::Emitter;

pub fn main() -> Result<()> {
    Emitter::default()
        .add_instructions(&Git2Builder::all_git()?)?
        .emit()
}
