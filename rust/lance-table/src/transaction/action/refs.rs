// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! References to counter-allocated ids, and the state that resolves them.

use crate::format::pb;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use std::collections::HashMap;

/// A reference to a counter-allocated identifier (field id, fragment id, or base
/// id) that may not be committed yet.
///
/// `Committed` is an already-assigned id. `Local` is a placeholder token minted by
/// an `Add*` action earlier in the same [`UserOperation`], resolved to a freshly
/// allocated id when the operation is applied.
///
/// Local tokens are scoped to one `UserOperation`, which is what makes minting
/// conflict-free: two concurrent operations each mint from the counter as they
/// apply, so their ids cannot collide however the tokens were numbered. It is
/// also what lets two branches that each add a field be merged — a placeholder
/// shared by several actions names the same not-yet-minted id, which a bare
/// sentinel value could not express.
///
/// [`UserOperation`]: super::UserOperation
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub enum Ref {
    Committed(u64),
    Local(u32),
}

impl TryFrom<pb::Ref> for Ref {
    type Error = Error;

    fn try_from(message: pb::Ref) -> Result<Self> {
        match message.kind {
            Some(pb::r#ref::Kind::Committed(id)) => Ok(Self::Committed(id)),
            Some(pb::r#ref::Kind::Local(token)) => Ok(Self::Local(token)),
            // Fail closed, for the reason given on `Action`'s decode: an
            // unrecognized reference kind must abort the commit rather than be
            // dropped from conflict detection.
            None => Err(Error::invalid_input(
                "Ref message did not contain a kind; it was written by a newer version of Lance",
            )),
        }
    }
}

impl From<&Ref> for pb::Ref {
    fn from(reference: &Ref) -> Self {
        let kind = match reference {
            Ref::Committed(id) => pb::r#ref::Kind::Committed(*id),
            Ref::Local(token) => pb::r#ref::Kind::Local(*token),
        };
        Self { kind: Some(kind) }
    }
}

/// State accumulated while applying one [`UserOperation`]'s actions in order.
///
/// A minting action records the id it allocated here, so a later action naming
/// the same `Local` token can resolve it. It grows a field per action family as
/// the vocabulary lands; today only base ids are minted.
///
/// There is deliberately no resolver yet: nothing in this slice *reads* a base
/// binding, and `AddDataFile` is the first action that will. Recording the
/// binding here is still not optional — minting without it would leave a seam
/// the next slice has to reopen, and it is where the "local tokens must be
/// distinct" rule that `actions.proto` promises is enforced at apply.
///
/// [`UserOperation`]: super::UserOperation
#[derive(Debug, Default)]
pub struct TxnContext {
    bases: HashMap<u32, u32>,
}

impl TxnContext {
    /// Record that `local` names the freshly minted base `id`.
    pub(crate) fn bind_base(&mut self, local: u32, id: u32) -> Result<()> {
        if let Some(existing) = self.bases.insert(local, id) {
            return Err(Error::invalid_input(format!(
                "local base token {local} is already bound to base id {existing}; \
                 local tokens must be distinct within a UserOperation"
            )));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn bound_base(&self, local: u32) -> Option<u32> {
        self.bases.get(&local).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ref_roundtrips() {
        for reference in [Ref::Committed(u64::MAX), Ref::Local(7)] {
            let message = pb::Ref::from(&reference);
            assert_eq!(Ref::try_from(message).unwrap(), reference);
        }
    }

    #[test]
    fn test_ref_without_kind_is_rejected() {
        let err = Ref::try_from(pb::Ref { kind: None }).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
        assert!(err.to_string().contains("did not contain a kind"));
    }

    #[test]
    fn test_bind_records_the_minted_id_under_its_token() {
        let mut ctx = TxnContext::default();
        ctx.bind_base(3, 9).unwrap();
        ctx.bind_base(4, 10).unwrap();
        assert_eq!(ctx.bound_base(3), Some(9));
        assert_eq!(ctx.bound_base(4), Some(10));
        assert_eq!(ctx.bound_base(5), None);
    }

    #[test]
    fn test_duplicate_local_token_is_rejected() {
        let mut ctx = TxnContext::default();
        ctx.bind_base(0, 1).unwrap();
        let err = ctx.bind_base(0, 2).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
        assert!(err.to_string().contains("already bound to base id 1"));
    }

    #[test]
    fn test_unbound_token_is_absent() {
        assert_eq!(TxnContext::default().bound_base(4), None);
    }
}
