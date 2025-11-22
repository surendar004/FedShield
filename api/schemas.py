"""Validation schemas for API requests."""
from marshmallow import Schema, fields, validate, ValidationError
from typing import Optional

class ThreatReportSchema(Schema):
    """Schema for threat report POST requests."""
    client_id = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=50),
        error_messages={'required': 'client_id is required', 'invalid': 'client_id must be a string'}
    )
    cpu_pct = fields.Float(
        required=True,
        validate=validate.Range(min=0.0, max=100.0),
        error_messages={'required': 'cpu_pct is required', 'invalid': 'cpu_pct must be between 0 and 100'}
    )
    net_bytes = fields.Float(
        required=True,
        validate=validate.Range(min=0.0),
        error_messages={'required': 'net_bytes is required', 'invalid': 'net_bytes must be non-negative'}
    )
    file_access_count = fields.Int(
        required=True,
        validate=validate.Range(min=0),
        error_messages={'required': 'file_access_count is required', 'invalid': 'file_access_count must be non-negative'}
    )
    file_path = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=500),
        error_messages={'required': 'file_path is required', 'invalid': 'file_path must be a string'}
    )
    is_threat = fields.Bool(
        load_default=False,
        error_messages={'invalid': 'is_threat must be a boolean'}
    )
    action = fields.Str(
        load_default='none',
        validate=validate.OneOf(['none', 'quarantine', 'quarantine_failed']),
        error_messages={'invalid': 'action must be one of: none, quarantine, quarantine_failed'}
    )
    timestamp = fields.Str(
        load_default=None,
        allow_none=True,
        error_messages={'invalid': 'timestamp must be a string'}
    )
    quarantined_path = fields.Str(
        load_default=None,
        allow_none=True,
        error_messages={'invalid': 'quarantined_path must be a string'}
    )


class ThreatQuerySchema(Schema):
    """Schema for threat query GET requests."""
    page = fields.Int(
        load_default=1,
        validate=validate.Range(min=1),
        error_messages={'invalid': 'page must be a positive integer'}
    )
    per_page = fields.Int(
        load_default=50,
        validate=validate.Range(min=1, max=1000),
        error_messages={'invalid': 'per_page must be between 1 and 1000'}
    )
    client_id = fields.Str(
        load_default=None,
        allow_none=True,
        validate=validate.Length(max=50),
        error_messages={'invalid': 'client_id must be a string'}
    )
    is_threat = fields.Bool(
        load_default=None,
        allow_none=True,
        error_messages={'invalid': 'is_threat must be a boolean'}
    )

