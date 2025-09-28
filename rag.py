
from pathlib import Path
from pypdf import PdfReader
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import re, json, uuid, math
from collections import Counter, defaultdict
