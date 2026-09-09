import aws_cdk as core
import aws_cdk.assertions as assertions

from my_cdk_app.infra_stack import InfraStack


def test_conversation_table_created():
    app = core.App()
    stack = InfraStack(app, "TestInfraStack")
    template = assertions.Template.from_stack(stack)

    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {"KeySchema": [
            {"AttributeName": "session_id", "KeyType": "HASH"},
            {"AttributeName": "turn_number", "KeyType": "RANGE"},
        ]},
    )