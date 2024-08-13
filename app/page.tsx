'use client';

import Head from 'next/head';
import Link from 'next/link';
import Image from 'next/image';
import React, { useState } from 'react';
import FileUpload from './FileUpload';

export default function Home() {
  const [uploadedFilename, setUploadedFilename] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);

  const handleFileUpload = (filename) => {
    setUploadedFilename(filename);
    setAnalyzed(false);
  };

  const handleFileRemove = () => {
    setUploadedFilename(null);
    setAnalyzed(false);
  };

  // TODO erase mock and call backend
  const handleSubmit = async () => {
    if (!uploadedFilename) return;

    setLoading(true);
    try {
      // Mock server response
      setTimeout(() => {
        const mockResponse = {
          skills: ['JavaScript', 'React', 'Node.js'],
          occupations: ['Frontend Developer', 'Full Stack Developer', 'Software Engineer']
        };
        setAnalyzed(true);
        setLoading(false);
      }, 2000);
    } catch (error) {
      console.error('Error:', error);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <Head>
        {/* TODO add title and logo */}
      </Head>

      <header className="flex justify-between items-center p-4 bg-white shadow">
        <div>
          <Image src="/media/3amk_full.png" alt="Logo" className="h-15" />
        </div>
        <nav className="flex space-x-4">
          <Link href="/about" className="text-gray-700 hover:text-gray-900">
            {/* TODO create page  */}
            About
          </Link>
          <Link href="/contact" className="text-gray-700 hover:text-gray-900">
            {/* TODO create page  */}
            Contact
          </Link>
        </nav>
      </header>

      <main className="container mx-auto px-4 py-16">
        <h1 className="text-5xl font-bold text-center mb-4">CareerBot</h1>
        <p className="text-xl text-center mb-8">Analyze your resume and build your profile !</p>

        <div className="flex flex-col items-center">
          <div className="w-full max-w-md">
            <FileUpload onFileUpload={handleFileUpload} onFileRemove={handleFileRemove} />
          </div>

          {uploadedFilename && (
            <button
              className="bg-blue-500 text-white px-4 py-2 mt-4 rounded shadow hover:bg-blue-600 w-full max-w-md"
              onClick={handleSubmit}
              disabled={loading || analyzed}
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          )}
        </div>

        {analyzed && (
          <div className="mt-8">
            {/* TODO catch back response and display data  */}
            <h2 className="text-2xl font-semibold text-center">Future backend response</h2>
          </div>
        )}
      </main>
    </div>
  );
}
