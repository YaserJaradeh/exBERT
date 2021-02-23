from paperswithcode import PapersWithCodeClient

client = PapersWithCodeClient()

if __name__ == '__main__':
    papers = client.paper_list()
    print(papers.results[0])
    print(papers.next_page)
