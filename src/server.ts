import express, { Request, Response } from 'express';
import multer from 'multer';
import cors from 'cors';
import fs from 'fs/promises';
import dotenv from 'dotenv';
import swaggerJsdoc from 'swagger-jsdoc';
import swaggerUi from 'swagger-ui-express';
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { ChatOpenAI } from '@langchain/openai';

// Load environment variables
dotenv.config();

// Swagger configuration
const swaggerOptions = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'RAG Chatbot API',
      version: '1.0.0',
      description: 'A RESTful API for RAG (Retrieval-Augmented Generation) Chatbot with FAISS vector store',
      contact: {
        name: 'API Support',
        email: 'support@example.com'
      }
    },
    servers: [
      {
        url: `http://localhost:${process.env.PORT || 3001}`,
        description: 'Development server'
      }
    ],
    components: {
      schemas: {
        Error: {
          type: 'object',
          properties: {
            error: {
              type: 'string',
              description: 'Error message'
            }
          }
        },
        ServerStatus: {
          type: 'object',
          properties: {
            message: {
              type: 'string'
            },
            timestamp: {
              type: 'string',
              format: 'date-time'
            },
            environment: {
              type: 'string'
            },
            vectorStoreLoaded: {
              type: 'boolean'
            }
          }
        },
        ApiStatus: {
          type: 'object',
          properties: {
            status: {
              type: 'string'
            },
            vectorStoreLoaded: {
              type: 'boolean'
            },
            apiKeys: {
              type: 'object',
              properties: {
                openai: {
                  type: 'boolean'
                },
                embeddingProvider: {
                  type: 'string'
                }
              }
            }
          }
        },
        UploadResponse: {
          type: 'object',
          properties: {
            message: {
              type: 'string'
            },
            filename: {
              type: 'string'
            }
          }
        },
        ChatRequest: {
          type: 'object',
          required: ['question'],
          properties: {
            question: {
              type: 'string',
              description: 'The question to ask about the uploaded documents'
            }
          }
        },
        ChatResponse: {
          type: 'object',
          properties: {
            answer: {
              type: 'string',
              description: 'AI-generated answer based on the document context'
            },
            sources: {
              type: 'integer',
              description: 'Number of relevant document chunks found'
            }
          }
        }
      }
    }
  },
  apis: ['./src/server.ts']
};

const specs = swaggerJsdoc(swaggerOptions);

const app = express();
const port = process.env.PORT || 3001;

// File upload configuration
const upload = multer({ dest: 'uploads/' });

// Constants
const FAISS_STORE_PATH = process.env.FAISS_INDEX_DIR || './faiss_index';
const TOP_K = parseInt(process.env.TOP_K || '4');

// Initialize embeddings based on provider
let embeddings: any;

if (process.env.EMBEDDING_PROVIDER === 'hf') {
  embeddings = new HuggingFaceInferenceEmbeddings({
    model: process.env.HF_EMBEDDING_MODEL || 'sentence-transformers/all-MiniLM-L6-v2',
  });
} else {
  // Fallback to OpenAI embeddings if not using HF
  const { OpenAIEmbeddings } = require('@langchain/openai');
  embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
  });
}

// Initialize OpenAI
const llm = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  modelName: process.env.OPENAI_MODEL || 'gpt-4o-mini',
  temperature: parseFloat(process.env.OPENAI_TEMPERATURE || '0.0'),
});

// Global vector store variable
let globalVectorStore: FaissStore | null = null;

// Middleware
app.use(cors());
app.use(express.json());

// Swagger UI
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(specs, {
  explorer: true,
  customCss: '.swagger-ui .topbar { display: none }',
  customSiteTitle: 'RAG Chatbot API Documentation'
}));

// FAISS utility functions
async function saveFaissStore(vectorStore: FaissStore): Promise<void> {
  try {
    await vectorStore.save(FAISS_STORE_PATH);
    console.log('✅ FAISS store saved to disk');
  } catch (error) {
    console.error('❌ Failed to save FAISS store:', error);
  }
}

async function loadFaissStore(): Promise<FaissStore | null> {
  try {
    const storeExists = await fs.access(FAISS_STORE_PATH).then(() => true).catch(() => false);

    if (storeExists) {
      const vectorStore = await FaissStore.load(FAISS_STORE_PATH, embeddings);
      console.log('✅ FAISS store loaded from disk');
      return vectorStore;
    } else {
      console.log('📁 No existing FAISS store found, creating new one');
      return null;
    }
  } catch (error) {
    console.error('❌ Failed to load FAISS store:', error);
    return null;
  }
}

async function processPdfAndStore(pdfPath: string): Promise<void> {
  try {
    // Create new vector store if none exists
    if (!globalVectorStore) {
      globalVectorStore = new FaissStore(embeddings, {});
    }

    // Load PDF
    const loader = new PDFLoader(pdfPath);
    const docs = await loader.load();
    console.log(`📄 Loaded PDF with ${docs.length} pages`);

    // Split into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const splitDocs = await textSplitter.splitDocuments(docs);
    console.log(`✂️ Split PDF into ${splitDocs.length} chunks`);

    // Add to vector store
    await globalVectorStore.addDocuments(splitDocs);
    console.log('🔗 Documents added to FAISS store');

    // Save to disk
    await saveFaissStore(globalVectorStore);

  } catch (error) {
    console.error('❌ Error processing PDF:', error);
    throw error;
  }
}

async function searchSimilarChunks(question: string, topK: number = 3): Promise<any[]> {
  if (!globalVectorStore) {
    throw new Error('No vector store available. Please upload a PDF first.');
  }

  try {
    const results = await globalVectorStore.similaritySearch(question, topK);
    console.log(`🔍 Found ${results.length} similar chunks for: "${question}"`);
    return results;
  } catch (error) {
    console.error('❌ Error during similarity search:', error);
    throw error;
  }
}

async function generateAnswer(question: string, context: any[]): Promise<string> {
  const contextText = context
    .map(doc => doc.pageContent)
    .join('\n\n');

  const prompt = `Based on the following context, answer the question. If the answer isn't in the context, say "I don't have enough information to answer that question."

Context:
${contextText}

Question: ${question}

Answer:`;

  try {
    const response = await llm.invoke(prompt);
    return response.content as string;
  } catch (error) {
    console.error('❌ Error generating answer:', error);
    throw error;
  }
}

// Routes

/**
 * @swagger
 * /:
 *   get:
 *     summary: Health check endpoint
 *     description: Returns server status and basic information
 *     tags:
 *       - Health
 *     responses:
 *       200:
 *         description: Server is running
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ServerStatus'
 */
app.get('/', (_req: Request, res: Response) => {
  res.json({
    message: 'RAG Chatbot Server is running!',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development',
    vectorStoreLoaded: globalVectorStore !== null
  });
});

/**
 * @swagger
 * /upload:
 *   post:
 *     summary: Upload and process PDF document
 *     description: Upload a PDF file to be processed and stored in the FAISS vector store for later querying
 *     tags:
 *       - Document Management
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             properties:
 *               pdf:
 *                 type: string
 *                 format: binary
 *                 description: PDF file to upload and process
 *     responses:
 *       200:
 *         description: PDF successfully processed and stored
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/UploadResponse'
 *       400:
 *         description: No PDF file uploaded
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *       500:
 *         description: Failed to process PDF
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 */
app.post('/upload', upload.single('pdf'), async (req: Request, res: Response) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No PDF file uploaded' });
    }

    console.log('📤 Processing uploaded PDF...');
    await processPdfAndStore(req.file.path);

    // Clean up uploaded file
    await fs.unlink(req.file.path);

    res.json({
      message: 'PDF processed and stored in FAISS successfully',
      filename: req.file.originalname
    });
  } catch (error) {
    console.error('❌ Upload error:', error);
    res.status(500).json({ error: 'Failed to process PDF' });
  }
});

/**
 * @swagger
 * /ask:
 *   post:
 *     summary: Ask a question about uploaded documents
 *     description: Submit a question to get an AI-generated answer based on the uploaded PDF documents using RAG (Retrieval-Augmented Generation)
 *     tags:
 *       - Chat
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/ChatRequest'
 *     responses:
 *       200:
 *         description: Successfully generated answer
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ChatResponse'
 *       400:
 *         description: Question is required
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *       500:
 *         description: Failed to generate answer
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 */
app.post('/ask', async (req: Request, res: Response) => {
  try {
    const { question } = req.body;

    if (!question) {
      return res.status(400).json({ error: 'Question is required' });
    }

    // Search for similar chunks
    const relevantChunks = await searchSimilarChunks(question, TOP_K);

    // Generate answer using Claude
    const answer = await generateAnswer(question, relevantChunks);

    res.json({
      answer,
      sources: relevantChunks.length
    });
  } catch (error) {
    console.error('❌ Ask error:', error);
    res.status(500).json({ error: 'Failed to generate answer' });
  }
});

/**
 * @swagger
 * /status:
 *   get:
 *     summary: Get API status and configuration
 *     description: Returns detailed status information including API key status, vector store status, and configuration details
 *     tags:
 *       - Health
 *     responses:
 *       200:
 *         description: API status information
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ApiStatus'
 */
app.get('/status', (_req: Request, res: Response) => {
  res.json({
    status: 'Server running',
    vectorStoreLoaded: globalVectorStore !== null,
    apiKeys: {
      openai: !!process.env.OPENAI_API_KEY,
      embeddingProvider: process.env.EMBEDDING_PROVIDER || 'openai'
    }
  });
});

// Initialize server
async function startServer() {
  try {
    // Try to load existing FAISS store on startup
    globalVectorStore = await loadFaissStore();

    app.listen(port, () => {
      console.log(`🚀 Server running on http://localhost:${port}`);
      console.log(`📁 Environment: ${process.env.NODE_ENV || 'development'}`);
      console.log(`🔑 API Keys: ${process.env.OPENAI_API_KEY ? '✅' : '❌'} OpenAI`);      console.log(`📊 Embedding Provider: ${process.env.EMBEDDING_PROVIDER || 'openai'}`);      console.log(`🎯 Model: ${process.env.OPENAI_MODEL || 'gpt-4o-mini'}`);      console.log(`🌡️ Temperature: ${process.env.OPENAI_TEMPERATURE || '0.0'}`);      console.log(`📈 Top K: ${TOP_K}`);
      console.log(`💾 Vector store loaded: ${globalVectorStore !== null ? '✅' : '❌'}`);
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
  }
}

startServer();