//! Pluggable event store abstraction for [`Store`](crate::Store).

use std::{error::Error, fmt, future::Future, pin::Pin, sync::Arc};

use eventsourcingdb::{
    Client, EventCandidate,
    error::ClientError,
    request_options::{EventType, ObserveEventsOptions, ReadEventsOptions},
};
use futures_util::{Stream, StreamExt};
use serde_json::Value;

/// A boxed future returned by an [`EventStore`] operation.
pub type EventStoreFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, EventStoreError>> + Send + 'a>>;

/// A boxed stream of stored events returned by [`EventStore::read_events`] and
/// [`EventStore::observe_events`].
pub type EventStream<'a> =
    Pin<Box<dyn Stream<Item = Result<eventsourcingdb::Event, EventStoreError>> + Send + 'a>>;

/// A boxed stream of registered event types returned by
/// [`EventStore::read_event_types`].
pub type EventTypeStream<'a> =
    Pin<Box<dyn Stream<Item = Result<EventType, EventStoreError>> + Send + 'a>>;

/// An error reported by an [`EventStore`].
///
/// The [`Client`] implementation wraps its [`ClientError`] directly in
/// [`EventStoreError::Client`], while custom event stores (such as an in-memory
/// or embedded database store) use the semantic variants to report precondition
/// conflicts, schema conflicts, and transient or permanent storage failures.
#[derive(Debug)]
pub enum EventStoreError {
    /// An error from the [`Client`] SDK.
    Client(Box<ClientError>),
    /// A write precondition did not hold, optionally naming the zero-based
    /// index of the violated precondition among the ones given to
    /// [`EventStore::write_events`].
    PreconditionViolated {
        /// The index of the violated precondition, if the event store knows
        /// which one failed.
        index: Option<usize>,
        /// A human-readable description of the precondition failure.
        description: String,
    },
    /// Registering a schema failed because the event type already has a
    /// schema, or because stored events of the type violate the schema.
    SchemaConflict {
        /// A human-readable description of the schema conflict.
        description: String,
    },
    /// A transient failure where retrying the same operation later may succeed.
    Transient(Box<dyn Error + Send + Sync + 'static>),
    /// A permanent failure where retrying the same operation will not help.
    Permanent(Box<dyn Error + Send + Sync + 'static>),
    /// A write or schema registration failed in a way that leaves open whether
    /// it was stored nonetheless.
    OutcomeUnknown(Box<dyn Error + Send + Sync + 'static>),
}

impl EventStoreError {
    /// Creates a permanent event store error with a plain message.
    #[must_use]
    pub fn permanent(message: impl Into<String>) -> Self {
        Self::Permanent(Box::new(MessageError(message.into())))
    }

    /// Creates a transient event store error with a plain message.
    #[must_use]
    pub fn transient(message: impl Into<String>) -> Self {
        Self::Transient(Box::new(MessageError(message.into())))
    }
}

impl fmt::Display for EventStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Client(error) => fmt::Display::fmt(error, formatter),
            Self::PreconditionViolated { description, .. }
            | Self::SchemaConflict { description } => formatter.write_str(description),
            Self::Transient(error) | Self::Permanent(error) | Self::OutcomeUnknown(error) => {
                fmt::Display::fmt(error, formatter)
            }
        }
    }
}

impl Error for EventStoreError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Client(error) => Some(error),
            Self::PreconditionViolated { .. } | Self::SchemaConflict { .. } => None,
            Self::Transient(error) | Self::Permanent(error) | Self::OutcomeUnknown(error) => {
                Some(&**error)
            }
        }
    }
}

impl From<ClientError> for EventStoreError {
    fn from(error: ClientError) -> Self {
        Self::Client(Box::new(error))
    }
}

/// A simple string error used by [`EventStoreError::permanent`] and
/// [`EventStoreError::transient`].
#[derive(Debug)]
struct MessageError(String);

impl fmt::Display for MessageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for MessageError {}

/// An underlying event store that [`Store`](crate::Store) reads from and
/// writes to.
///
/// [`Client`] implements this trait out of the box, and custom event stores
/// (such as an in-memory store or an embedded `SQLite` store on mobile or
/// desktop) can implement it to back a [`Store`](crate::Store).
pub trait EventStore: fmt::Debug + Send + Sync + 'static {
    /// Reads stored events for `subject` under `options`.
    fn read_events<'a, 'fut>(
        &'a self,
        subject: &'fut str,
        options: Option<ReadEventsOptions<'fut>>,
    ) -> EventStoreFuture<'fut, EventStream<'a>>
    where
        'a: 'fut;

    /// Observes stored and newly written events for `subject` under `options`.
    fn observe_events<'a, 'fut>(
        &'a self,
        subject: &'fut str,
        options: Option<ObserveEventsOptions<'fut>>,
    ) -> EventStoreFuture<'fut, EventStream<'a>>
    where
        'a: 'fut;

    /// Atomically checks `preconditions` and appends `candidates`, returning
    /// the stored events as recorded.
    fn write_events(
        &self,
        candidates: Vec<EventCandidate>,
        preconditions: Vec<eventsourcingdb::Precondition>,
    ) -> EventStoreFuture<'_, Vec<eventsourcingdb::Event>>;

    /// Reads all known event types and their registered schemas.
    fn read_event_types(&self) -> EventStoreFuture<'_, EventTypeStream<'_>>;

    /// Registers a JSON `schema` for `event_type`.
    fn register_event_schema<'a>(
        &'a self,
        event_type: &'a str,
        schema: &'a Value,
    ) -> EventStoreFuture<'a, ()>;
}

impl EventStore for Client {
    fn read_events<'a, 'fut>(
        &'a self,
        subject: &'fut str,
        options: Option<ReadEventsOptions<'fut>>,
    ) -> EventStoreFuture<'fut, EventStream<'a>>
    where
        'a: 'fut,
    {
        Box::pin(async move {
            let stream = Self::read_events(self, subject, options).await?;
            let mapped: EventStream<'a> =
                Box::pin(stream.map(|item| item.map_err(EventStoreError::from)));
            Ok(mapped)
        })
    }

    fn observe_events<'a, 'fut>(
        &'a self,
        subject: &'fut str,
        options: Option<ObserveEventsOptions<'fut>>,
    ) -> EventStoreFuture<'fut, EventStream<'a>>
    where
        'a: 'fut,
    {
        Box::pin(async move {
            let stream = Self::observe_events(self, subject, options).await?;
            let mapped: EventStream<'a> =
                Box::pin(stream.map(|item| item.map_err(EventStoreError::from)));
            Ok(mapped)
        })
    }

    fn write_events(
        &self,
        candidates: Vec<EventCandidate>,
        preconditions: Vec<eventsourcingdb::Precondition>,
    ) -> EventStoreFuture<'_, Vec<eventsourcingdb::Event>> {
        Box::pin(async move { Ok(Self::write_events(self, candidates, preconditions).await?) })
    }

    fn read_event_types(&self) -> EventStoreFuture<'_, EventTypeStream<'_>> {
        Box::pin(async move {
            let stream = Self::read_event_types(self).await?;
            let mapped: EventTypeStream<'_> =
                Box::pin(stream.map(|item| item.map_err(EventStoreError::from)));
            Ok(mapped)
        })
    }

    fn register_event_schema<'a>(
        &'a self,
        event_type: &'a str,
        schema: &'a Value,
    ) -> EventStoreFuture<'a, ()> {
        Box::pin(async move {
            let _ = Self::register_event_schema(self, event_type, schema).await?;
            Ok(())
        })
    }
}

impl<E: EventStore + ?Sized> EventStore for Arc<E> {
    fn read_events<'a, 'fut>(
        &'a self,
        subject: &'fut str,
        options: Option<ReadEventsOptions<'fut>>,
    ) -> EventStoreFuture<'fut, EventStream<'a>>
    where
        'a: 'fut,
    {
        (**self).read_events(subject, options)
    }

    fn observe_events<'a, 'fut>(
        &'a self,
        subject: &'fut str,
        options: Option<ObserveEventsOptions<'fut>>,
    ) -> EventStoreFuture<'fut, EventStream<'a>>
    where
        'a: 'fut,
    {
        (**self).observe_events(subject, options)
    }

    fn write_events(
        &self,
        candidates: Vec<EventCandidate>,
        preconditions: Vec<eventsourcingdb::Precondition>,
    ) -> EventStoreFuture<'_, Vec<eventsourcingdb::Event>> {
        (**self).write_events(candidates, preconditions)
    }

    fn read_event_types(&self) -> EventStoreFuture<'_, EventTypeStream<'_>> {
        (**self).read_event_types()
    }

    fn register_event_schema<'a>(
        &'a self,
        event_type: &'a str,
        schema: &'a Value,
    ) -> EventStoreFuture<'a, ()> {
        (**self).register_event_schema(event_type, schema)
    }
}
