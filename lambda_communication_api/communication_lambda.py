import json
import boto3
from botocore.exceptions import ClientError

# from flask import Flask, request, jsonify
from flask import jsonify

KEY_CHANNEL_ID = "channel_id"
KEY_DEVICE_ID = "device_id"
KEY_STYLE_NAME = "style_name"
KEY_NEED_TO_SAVE_PICTURES = "need_to_save_pictures"
KEY_NEW_STYLE_NAME = "new_style_name"
KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES = "additional_text_to_use_in_filenames"
DYNAMODB_RESOURCE = boto3.resource("dynamodb", region_name="eu-west-3")
COMMUNICATION_TABLE = DYNAMODB_RESOURCE.Table("inoft-portraits_device-communication")

# app = Flask(__name__)


# @app.route("/")
def get_all_infos_of_channel(request_parameters):
    if not isinstance(request_parameters, dict) or KEY_DEVICE_ID not in request_parameters.keys():
        return {"statusCode": 400,
                "body": json.dumps(f"Error, the {KEY_DEVICE_ID} attr was not found in the request_parameters.")}

    key_need_to_save_picture_with_device_id = f"{KEY_NEED_TO_SAVE_PICTURES}-{request_parameters[KEY_DEVICE_ID]}"

    try:
        channel_id = None
        if isinstance(request_parameters, dict) and KEY_CHANNEL_ID in request_parameters.keys():
            channel_id = request_parameters[KEY_CHANNEL_ID]
        if channel_id is not None:
            response = COMMUNICATION_TABLE.get_item(Key={KEY_CHANNEL_ID: channel_id})
        else:
            return {"statusCode": 401,
                    "body": json.dumps(f"The channel_id was not present in the request parameters.")}
    except ClientError as e:
        print(e.response["Error"]["Message"])
        return {"statusCode": 400,
                "body": json.dumps(f"Error when trying to access the communication table : {e.response['Error']['Message']}")}
    else:
        print(response)
        if isinstance(response, dict) and "Item" in response.keys():
            item = response["Item"]
            processed_item_object = dict()

            if isinstance(item, dict):
                if KEY_STYLE_NAME in item.keys():
                    current_style_name = item[KEY_STYLE_NAME]
                    try:
                        current_style_name = str(current_style_name)
                        processed_item_object[KEY_STYLE_NAME] = current_style_name
                    except Exception:
                        return {"statusCode": 400,
                                "body": json.dumps(f"The value at the {KEY_STYLE_NAME} key was not convertible to an str : {item[KEY_STYLE_NAME]}")}

                if key_need_to_save_picture_with_device_id in item.keys():
                    need_to_save_pictures = item[key_need_to_save_picture_with_device_id]
                    try:
                        need_to_save_pictures = bool(need_to_save_pictures)
                        processed_item_object[key_need_to_save_picture_with_device_id] = need_to_save_pictures
                    except Exception:
                        return {"statusCode": 400,
                                "body": json.dumps(f"The value at the {KEY_STYLE_NAME} key was not convertible to a bool : {item[KEY_STYLE_NAME]}")}

                if KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES in item.keys():
                    additional_text_to_use_in_filenames_object = item[KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES]
                    try:
                        additional_text_to_use_in_filenames_string = str(additional_text_to_use_in_filenames_object)
                        processed_item_object[
                            KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES] = additional_text_to_use_in_filenames_string
                    except Exception:
                        return {"statusCode": 400,
                                "body": json.dumps(f"The value at the {KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES} key was not convertible to an str : {item[KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES]}")}

                return {"statusCode": 200,
                        "body": json.dumps(processed_item_object)}

            else:
                return {"statusCode": 400,
                        "body": json.dumps(f"The received item was not in the form of a dict")}
        else:
            return {"statusCode": 400,
                    "body": json.dumps(f"No item has been received from the response of the database.")}


# @app.route("/style-name")
def get_style_name(request_path, request_method, request_parameters):
    try:
        if isinstance(request_parameters, dict) and KEY_CHANNEL_ID in request_parameters.keys():
            channel_id = request_parameters[KEY_CHANNEL_ID]
            response = COMMUNICATION_TABLE.get_item(Key={KEY_CHANNEL_ID: channel_id})
        else:
            return {"statusCode": 401,
                    "body": json.dumps(f"The channel_id was not present in the request parameters.")}
    except ClientError as e:
        print(e.response["Error"]["Message"])
        return {"statusCode": 401,
                "body": json.dumps(
                    f"Error when trying to access the communication table : {e.response['Error']['Message']}")}
    else:
        print(response)
        if isinstance(response, dict) and "Item" in response.keys():
            item = response["Item"]

            if isinstance(item, dict) and KEY_STYLE_NAME in item.keys():
                current_style_name = item[KEY_STYLE_NAME]

                # We do a try except to see if we can convert the received
                # style index to an str, otherwise, it is an invalid value.
                try:
                    current_style_name = str(current_style_name)
                    return {"statusCode": 200,
                            "body": json.dumps(current_style_name)}

                except Exception:
                    return {"statusCode": 401,
                            "body": json.dumps(
                                f"The value at the {KEY_STYLE_NAME} key was not convertible to an str : {item[KEY_STYLE_NAME]}")}
            else:
                return {"statusCode": 401,
                        "body": json.dumps(
                            f"The received item was not in the form of a dict or did not contained a {KEY_STYLE_NAME} key : {item}")}
        else:
            return {"statusCode": 401,
                    "body": json.dumps(f"No item has been received from the response of the database.")}


# @app.route("/style-name", methods=["POST"])
def change_style_name(request_path, request_method, request_parameters):
    if isinstance(request_parameters, dict) and KEY_CHANNEL_ID in request_parameters.keys() and KEY_NEW_STYLE_NAME in request_parameters.keys():
        channel_id = request_parameters[KEY_CHANNEL_ID]
        print(channel_id)
        new_style_name = request_parameters[KEY_NEW_STYLE_NAME]

        try:
            response = COMMUNICATION_TABLE.update_item(
                Key={
                    "channel_id": channel_id,
                },
                UpdateExpression=f"set style_name = :new_style_name",
                ConditionExpression="channel_id = :channel_id",
                ExpressionAttributeValues={
                    ":new_style_name": new_style_name,
                    ':channel_id': channel_id,
                },
                ReturnValues="UPDATED_NEW"
            )
            return {"statusCode": 200, "body": json.dumps(new_style_name)}
        except ClientError as e:
            if e.response['Error']['Code'] == "ConditionalCheckFailedException":
                print(e.response['Error']['Message'])
                print(f"L'id est invalid.")
                return {"statusCode": 403,
                        "body": json.dumps(f"The channel_id was not valid.")}
            else:
                return {"statusCode": 401,
                        "body": json.dumps(f"Unknow error while accessing the communication dynamodb table.")}
    else:
        return {"statusCode": 401,
                "body": json.dumps(f"The channel_id or the new_style_name was not present in the request parameters.")}


# @app.route("/save-pictures")
def set_save_pictures(request_parameters, new_value_need_to_save_pictures: bool):
    if isinstance(request_parameters, dict) and KEY_CHANNEL_ID in request_parameters.keys():
        channel_id = request_parameters[KEY_CHANNEL_ID]
        print(channel_id)

        update_expression = ""
        if KEY_DEVICE_ID in request_parameters.keys():
            update_expression += f"set {KEY_NEED_TO_SAVE_PICTURES}-{request_parameters[KEY_DEVICE_ID]} = :value_need_to_save_pictures"
        else:
            update_expression += f"set {KEY_NEED_TO_SAVE_PICTURES}-0 = :value_need_to_save_pictures, " \
                                 f"set {KEY_NEED_TO_SAVE_PICTURES}-1 = :value_need_to_save_pictures"


        additional_text_to_use_in_filenames_value = " "  # DynamoDB cannot take a totally empty string
        if KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES in request_parameters.keys():
            additional_text_to_use_in_filenames_value = request_parameters[KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES]

        try:
            response = COMMUNICATION_TABLE.update_item(
                Key={
                    "channel_id": channel_id,
                },
                UpdateExpression=f"{update_expression}, {KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES} = :additional_text_to_use_in_filenames",

                ConditionExpression="channel_id = :channel_id",
                ExpressionAttributeValues={
                    ":value_need_to_save_pictures": new_value_need_to_save_pictures,
                    ":additional_text_to_use_in_filenames": additional_text_to_use_in_filenames_value,
                    ':channel_id': channel_id,
                },
                ReturnValues="UPDATED_NEW"
            )
            return {"statusCode": 200, "body": json.dumps(f"Need to save pictures of {channel_id} set to {new_value}")}
        except ClientError as e:
            if e.response['Error']['Code'] == "ConditionalCheckFailedException":
                print(e.response['Error']['Message'])
                print(f"L'id est invalid.")
                return {"statusCode": 403,
                        "body": json.dumps(f"The channel_id was not valid.")}
            else:
                return {"statusCode": 401,
                        "body": json.dumps(f"Unknow error while accessing the communication dynamodb table.")}
    else:
        return {"statusCode": 401,
                "body": json.dumps(f"The channel_id was not present in the request parameters.")}


def lambda_handler(event, context):
    print(f"event = {event}")

    request_path = event["path"]
    request_method = event["httpMethod"]
    request_parameters = event["queryStringParameters"]

    if request_path == "/":
        return get_all_infos_of_channel(request_parameters=request_parameters)
    elif request_path == "/style-name":
        if request_method == "GET":
            return get_style_name(request_path=request_path, request_method=request_method,
                                  request_parameters=request_parameters)
        elif request_method == "POST":
            return change_style_name(request_path=request_path, request_method=request_method,
                                     request_parameters=request_parameters)
    elif request_path == "/save-pictures":
        if request_method == "POST":
            return set_save_pictures(request_parameters=request_parameters, new_value_need_to_save_pictures=True)
        elif request_method == "DELETE":
            return set_save_pictures(request_parameters=request_parameters, new_value_need_to_save_pictures=False)

