from tool_contracts import ToolContract, ToolRegistry


def book_flight(flight_id, passengers):
    return {"booked": flight_id, "passengers": passengers}


BROKEN = ToolContract(
    name="book flight",
    description="books",
    input_schema={
        "type": "object",
        "properties": {
            "flight_id": {"type": "string", "pattern": "^[A-C]$"},
            "passengers": {"type": "integer", "minimum": 1},
        },
    },
    handler=book_flight,
)

registry = ToolRegistry([BROKEN])
