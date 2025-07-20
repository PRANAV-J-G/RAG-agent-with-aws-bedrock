import os
import json
import boto3
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session
import logging
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv
from time import time 

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'shellkode-rag-documents')
BEDROCK_AGENT_ID = os.getenv('BEDROCK_AGENT_ID', 'AMA9OYJSAA')
BEDROCK_AGENT_ALIAS_ID = os.getenv('BEDROCK_AGENT_ALIAS_ID', 'V0XBFRQZUC')
KNOWLEDGE_BASE_ID = os.getenv('KNOWLEDGE_BASE_ID', 'VVU0EDVBWU')
DATA_SOURCE_ID = os.getenv('DATA_SOURCE_ID', 'R7GVAC04R2')

# Global variables for AWS clients
s3_client = None
bedrock_agent_runtime = None
bedrock_agent = None
aws_connection_status = False
connection_error_message = ""

def initialize_aws_clients():
    """Initialize AWS clients with comprehensive error handling"""
    global s3_client, bedrock_agent_runtime, bedrock_agent, aws_connection_status, connection_error_message
    
    try:
        # Validate environment variables
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            error_msg = "AWS credentials not found in environment variables"
            logger.error(error_msg)
            connection_error_message = error_msg
            return False
        
        # Clean credentials
        access_key = str(AWS_ACCESS_KEY_ID).strip().strip('"').strip("'")
        secret_key = str(AWS_SECRET_ACCESS_KEY).strip().strip('"').strip("'")
        
        logger.info(f" Creating AWS session with access key: {access_key[:8]}...")
        
        # Create session with explicit credentials
        session_aws = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=AWS_REGION
        )
        
        # Test credentials with STS
        logger.info(" Testing AWS credentials with STS...")
        sts_client = session_aws.client('sts')
        identity = sts_client.get_caller_identity()
        logger.info(f" AWS credentials validated successfully!")
        logger.info(f" Account: {identity.get('Account')}")
        logger.info(f" User ARN: {identity.get('Arn')}")
        
        # Create service clients
        logger.info("🛠️ Creating AWS service clients...")
        s3_client = session_aws.client('s3')
        bedrock_agent_runtime = session_aws.client('bedrock-agent-runtime')
        bedrock_agent = session_aws.client('bedrock-agent')
        
        # Test S3 access
        logger.info(f"Testing S3 bucket access: {S3_BUCKET_NAME}")
        try:
            s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
            logger.info(" S3 bucket access confirmed")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                logger.warning(f" S3 bucket '{S3_BUCKET_NAME}' may not exist, but credentials are valid")
            else:
                logger.warning(f" S3 bucket access issue: {e.response['Error']['Code']}")
        
        # Test Bedrock access
        logger.info(" Testing Bedrock agent access...")
        try:
            bedrock_agent.get_agent(agentId=BEDROCK_AGENT_ID)
            logger.info(" Bedrock agent access confirmed")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.warning(f" Bedrock agent '{BEDROCK_AGENT_ID}' may not exist, but credentials are valid")
            else:
                logger.warning(f" Bedrock access issue: {e.response['Error']['Code']}")
        
        aws_connection_status = True
        connection_error_message = ""
        logger.info("All AWS clients initialized successfully!")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error'].get('Message', '')
        
        if error_code == 'InvalidClientTokenId':
            error_msg = f" Invalid AWS Access Key ID: {access_key[:8]}... Please verify the access key is correct and active."
        elif error_code == 'SignatureDoesNotMatch':
            error_msg = f" Invalid AWS Secret Access Key. Please verify the secret key is correct."
        elif error_code == 'TokenRefreshRequired':
            error_msg = f" AWS token refresh required. Please create new access keys."
        elif error_code == 'AccessDenied':
            error_msg = f" Access denied for {e.operation_name}. User needs additional IAM permissions."
        else:
            error_msg = f"AWS Error ({error_code}): {error_message}"
        
        logger.error(f"AWS ClientError: {error_msg}")
        connection_error_message = error_msg
        return False
        
    except (NoCredentialsError, PartialCredentialsError) as e:
        error_msg = f"AWS credentials configuration error: {str(e)}"
        logger.error(error_msg)
        connection_error_message = error_msg
        return False
        
    except Exception as e:
        error_msg = f"Unexpected error initializing AWS clients: {str(e)}"
        logger.error(error_msg)
        connection_error_message = error_msg
        return False

class CookingRAGSystem:
    def __init__(self):
        self.session_id = None
    
    def check_aws_connection(self):
        """Check if AWS clients are properly initialized"""
        if not aws_connection_status:
            return False, connection_error_message or "AWS clients not initialized"
        if not all([s3_client, bedrock_agent_runtime, bedrock_agent]):
            return False, "AWS clients not properly configured"
        return True, "AWS connection OK"
    
    def query_cooking_agent(self, query, session_id=None):
        """Query the Bedrock agent for cooking-related questions"""
        try:
            # Check AWS connection first
            is_connected, message = self.check_aws_connection()
            if not is_connected:
                return {
                    'response': f"🚫 Cannot process your cooking query: {message}",
                    'session_id': session_id,
                    'success': False
                }
            
            if not session_id:
                session_id = str(uuid.uuid4())
            
            logger.info(f"🍳 Processing cooking query with session: {session_id[:8]}...")
            
            # First attempt with current session - send query directly
            try:
                response = bedrock_agent_runtime.invoke_agent(
                    agentId=BEDROCK_AGENT_ID,
                    agentAliasId=BEDROCK_AGENT_ALIAS_ID,
                    sessionId=session_id,
                    inputText=query  # Send user query directly - agent instructions handle the rest
                )
                
                # Process the streaming response
                result = ""
                for event in response['completion']:
                    if 'chunk' in event:
                        chunk = event['chunk']
                        if 'bytes' in chunk:
                            result += chunk['bytes'].decode('utf-8')
                
                if result.strip():
                    logger.info("✅ Cooking agent query completed successfully")
                    return {
                        'response': result.strip(),
                        'session_id': session_id,
                        'success': True
                    }
                    
            except ClientError as context_error:
                if 'context window' in str(context_error).lower() or 'memory turns' in str(context_error).lower():
                    logger.warning(f"⚠️ Context window exceeded for session {session_id[:8]}, creating new session...")
                    
                    # Create new session and retry
                    new_session_id = str(uuid.uuid4())
                    logger.info(f"🔄 Retrying with new session: {new_session_id[:8]}...")
                    
                    try:
                        response = bedrock_agent_runtime.invoke_agent(
                            agentId=BEDROCK_AGENT_ID,
                            agentAliasId=BEDROCK_AGENT_ALIAS_ID,
                            sessionId=new_session_id,
                            inputText=query  # Send user query directly
                        )
                        
                        # Process the streaming response
                        result = ""
                        for event in response['completion']:
                            if 'chunk' in event:
                                chunk = event['chunk']
                                if 'bytes' in chunk:
                                    result += chunk['bytes'].decode('utf-8')
                        
                        if result.strip():
                            logger.info("✅ Cooking agent query completed with new session")
                            return {
                                'response': result.strip(),
                                'session_id': new_session_id,
                                'success': True,
                                'new_session': True  # Flag to update frontend session
                            }
                    except Exception as retry_error:
                        logger.error(f"Retry with new session failed: {str(retry_error)}")
                        raise context_error  # Re-raise original error
                else:
                    raise context_error  # Re-raise if not context window issue
            
            # If no result from either attempt
            result = """🍳 I apologize, but I couldn't find specific information about that recipe or cooking technique in my knowledge base. 

Here are some ways I can help you:
- Ask about specific recipes (e.g., "How do I make chocolate chip cookies?")
- Cooking techniques (e.g., "How to properly sauté vegetables?")
- Ingredient substitutions (e.g., "What can I use instead of eggs in baking?")
- Troubleshooting cooking problems

Please try rephrasing your question or ask about a specific recipe or cooking technique!"""
            
            return {
                'response': result,
                'session_id': session_id,
                'success': True
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error'].get('Message', '')
            
            if error_code == 'ResourceNotFoundException':
                response_msg = f" Bedrock agent not found. Please verify agent ID: {BEDROCK_AGENT_ID}"
            elif error_code == 'AccessDeniedException':
                response_msg = "Access denied to Bedrock agent. Check IAM permissions."
            elif 'context window' in error_message.lower() or 'memory turns' in error_message.lower():
                response_msg = "Conversation too long. Starting fresh session..."
                # Return with new session flag
                return {
                    'response': response_msg,
                    'session_id': str(uuid.uuid4()),
                    'success': True,
                    'new_session': True
                }
            else:
                response_msg = f" Bedrock error: {error_message}"
            
            logger.error(f"Cooking agent query error: {response_msg}")
            return {
                'response': response_msg,
                'session_id': session_id,
                'success': False
            }
            
        except Exception as e:
            logger.error(f"Error querying cooking agent: {str(e)}")
            return {
                'response': f"I encountered an error while processing your cooking query: {str(e)}",
                'session_id': session_id,
                'success': False
            }

    def get_recipe_suggestions(self):
        """Get sample recipe suggestions with variety"""
        suggestions = [
            "Tell me something about south indian cuisine",
            "What are the ingredients to make biriyani??",
            "Tell me something about south indian cooking",
            "Tell me something about jackfruit leather",
            "What do you know about Pakistani/ Mughlai Cuisines??",
            "What's a good vegetarian dinner recipe?",
            "Can you tell me something about fundamentals of cooking",
            "What are the cooking methods involved in arabian cuisine?",
            "What are the ingredients required to make South Indian Dosa?",
            "What are some healthy breakfast ideas?",
            "How do I make pizza dough at home?",
            "What's the secret to perfect scrambled eggs?"
        ]
        # Return random 8 suggestions each time for variety
        import random
        return random.sample(suggestions, 8)

# Initialize Cooking RAG system
cooking_rag = CookingRAGSystem()

# Try to initialize AWS clients on startup
logger.info("Initializing AWS clients for CookingGenie...")
aws_initialized = initialize_aws_clients()

if not aws_initialized:
    logger.warning("AWS clients failed to initialize. Application will run in limited mode.")
    logger.warning(f"Error: {connection_error_message}")

@app.route('/')
def index():
    """Main page with cooking chat interface"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_cooking():
    """Handle cooking-related queries"""
    try:
        # Check if AWS is connected first
        if not aws_connection_status:
            return jsonify({
                'success': False, 
                'message': f'Cannot process cooking query: {connection_error_message}'
            })
        
        data = request.get_json()
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return jsonify({'success': False, 'message': '🍳 Please ask me a cooking question!'})
        
        logger.info(f"🍳 Processing cooking query: {query_text[:50]}...")
        
        # Get or create session ID
        session_id = session.get('cooking_session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['cooking_session_id'] = session_id
        
        # Query the cooking agent
        result = cooking_rag.query_cooking_agent(query_text, session_id)
        
        # Check if we got a new session due to context window limits
        if result.get('new_session', False):
            session['cooking_session_id'] = result['session_id']
            logger.info(f" Updated session ID to: {result['session_id'][:8]}...")
        
        if result['success']:
            logger.info("Cooking query processed successfully")
        else:
            logger.warning(f" Cooking query processing had issues: {result['response']}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Cooking query error: {str(e)}")
        return jsonify({'success': False, 'message': f'🚨 Cooking query failed: {str(e)}'})

@app.route('/suggestions')
def get_suggestions():
    """Get cooking recipe suggestions"""
    try:
        suggestions = cooking_rag.get_recipe_suggestions()
        return jsonify({'success': True, 'suggestions': suggestions})
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        return jsonify({'success': False, 'suggestions': []})

@app.route('/health')
def health_check():
    """Health check endpoint with detailed AWS status"""
    try:
        status = {
            'status': 'healthy' if aws_connection_status else 'degraded',
            'service': 'CookingGenie RAG System',
            'timestamp': datetime.now().isoformat(),
            'aws_connected': aws_connection_status,
            'credentials_loaded': bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY),
            'services': {}
        }
        
        if aws_connection_status:
            # Test each service
            try:
                s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
                status['services']['s3'] = 'connected'
            except Exception as e:
                status['services']['s3'] = f'error: {str(e)}'
            
            status['services']['bedrock_agent'] = 'available' if bedrock_agent else '❌ not_initialized'
            status['services']['bedrock_runtime'] = 'available' if bedrock_agent_runtime else '❌ not_initialized'
            status['bucket_name'] = S3_BUCKET_NAME
            status['agent_id'] = BEDROCK_AGENT_ID
            status['knowledge_base_id'] = KNOWLEDGE_BASE_ID
        else:
            status['error'] = connection_error_message
            status['services'] = {'s3': 'disconnected', 'bedrock': 'disconnected'}
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy', 
            'service': 'CookingGenie RAG System',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/reinitialize-aws', methods=['POST'])
def reinitialize_aws():
    """Endpoint to retry AWS initialization"""
    global aws_connection_status, connection_error_message
    
    logger.info("Attempting to reinitialize AWS clients...")
    success = initialize_aws_clients()
    
    if success:
        return jsonify({'success': True, 'message': 'AWS clients reinitialized successfully'})
    else:
        return jsonify({'success': False, 'message': connection_error_message})

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'message': 'Endpoint not found'}), 404

if __name__ == '__main__':
    # Print startup status
    print("\n" + "="*60)
    print("CookingGenie RAG Assistant Starting...")
    print(f"AWS Connection: {'Connected' if aws_connection_status else 'Failed'}")
    if not aws_connection_status:
        print(f"Error: {connection_error_message}")
        print("Check your AWS credentials and try /reinitialize-aws endpoint")
    else:
        print(f"S3 Bucket: {S3_BUCKET_NAME}")
        print(f"Bedrock Agent: {BEDROCK_AGENT_ID}")
        print(f"Knowledge Base: {KNOWLEDGE_BASE_ID}")
    print("="*60 + "\n")
    
    # Run the application
    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)